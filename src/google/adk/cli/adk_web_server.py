# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import importlib
import json
import logging
import os
import re
import sys
import time
import traceback
import typing
from typing import Any
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional

from fastapi import Body
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Path
from fastapi import Query
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket
from fastapi.websockets import WebSocketDisconnect
from google.genai import types
import graphviz
from opentelemetry import trace
import opentelemetry.sdk.environment_variables as otel_env
from opentelemetry.sdk.trace import export as export_lib
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from pydantic import Field
from pydantic import ValidationError
from starlette.types import Lifespan
from typing_extensions import deprecated
from typing_extensions import override
from watchdog.observers import Observer

from . import agent_graph
from ..agents.base_agent import BaseAgent
from ..agents.live_request_queue import LiveRequest
from ..agents.live_request_queue import LiveRequestQueue
from ..agents.run_config import RunConfig
from ..agents.run_config import StreamingMode
from ..apps.app import App
from ..artifacts.base_artifact_service import ArtifactVersion
from ..artifacts.base_artifact_service import BaseArtifactService
from ..auth.credential_service.base_credential_service import BaseCredentialService
from ..errors.already_exists_error import AlreadyExistsError
from ..errors.input_validation_error import InputValidationError
from ..errors.not_found_error import NotFoundError
from ..errors.session_not_found_error import SessionNotFoundError
from ..evaluation.base_eval_service import InferenceConfig
from ..evaluation.base_eval_service import InferenceRequest
from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from ..evaluation.eval_case import EvalCase
from ..evaluation.eval_case import SessionInput
from ..evaluation.eval_metrics import EvalMetric
from ..evaluation.eval_metrics import EvalMetricResult
from ..evaluation.eval_metrics import EvalMetricResultPerInvocation
from ..evaluation.eval_metrics import EvalStatus
from ..evaluation.eval_metrics import MetricInfo
from ..evaluation.eval_result import EvalSetResult
from ..evaluation.eval_set import EvalSet
from ..evaluation.eval_set_results_manager import EvalSetResultsManager
from ..evaluation.eval_sets_manager import EvalSetsManager
from ..events.event import Event
from ..memory.base_memory_service import BaseMemoryService
from ..plugins.base_plugin import BasePlugin
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from ..utils.context_utils import Aclosing
from ..version import __version__
from .cli_eval import EVAL_SESSION_ID_PREFIX
from .utils import cleanup
from .utils import common
from .utils import envs
from .utils import evals
from .utils.base_agent_loader import BaseAgentLoader
from .utils.shared_value import SharedValue
from .utils.state import create_empty_state

logger = logging.getLogger("google_adk." + __name__)

_EVAL_SET_FILE_EXTENSION = ".evalset.json"

TAG_DEBUG = "Debug"
TAG_EVALUATION = "Evaluation"
TAG_SESSIONS = "Sessions"
TAG_RUNS = "Agent Runs"
TAG_ARTIFACTS = "Artifacts"
TAG_MEMORY = "Memory"

_REGEX_PREFIX = "regex:"


def _parse_cors_origins(
    allow_origins: list[str],
) -> tuple[list[str], Optional[str]]:
  """Parse allow_origins into literal origins and a combined regex pattern.

  Args:
    allow_origins: List of origin strings. Entries prefixed with 'regex:' are
      treated as regex patterns; all others are treated as literal origins.

  Returns:
    A tuple of (literal_origins, combined_regex) where combined_regex is None
    if no regex patterns were provided, or a single pattern joining all regex
    patterns with '|'.
  """
  literal_origins = []
  regex_patterns = []
  for origin in allow_origins:
    if origin.startswith(_REGEX_PREFIX):
      pattern = origin[len(_REGEX_PREFIX) :]
      if pattern:
        regex_patterns.append(pattern)
    else:
      literal_origins.append(origin)

  combined_regex = "|".join(regex_patterns) if regex_patterns else None
  return literal_origins, combined_regex


def _is_origin_allowed(
    origin: str,
    allowed_literal_origins: list[str],
    allowed_origin_regex: Optional[re.Pattern[str]],
) -> bool:
  """Check whether the given origin matches the allowed origins."""
  if "*" in allowed_literal_origins:
    return True
  if origin in allowed_literal_origins:
    return True
  if allowed_origin_regex is not None:
    return allowed_origin_regex.fullmatch(origin) is not None
  return False


def _normalize_origin_scheme(scheme: str) -> str:
  """Normalize request schemes to the browser Origin scheme space."""
  if scheme == "ws":
    return "http"
  if scheme == "wss":
    return "https"
  return scheme


def _strip_optional_quotes(value: str) -> str:
  """Strip a single pair of wrapping quotes from a header value."""
  if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
    return value[1:-1]
  return value


def _get_scope_header(
    scope: dict[str, Any], header_name: bytes
) -> Optional[str]:
  """Return the first matching header value from an ASGI scope."""
  for candidate_name, candidate_value in scope.get("headers", []):
    if candidate_name == header_name:
      return candidate_value.decode("latin-1").split(",", 1)[0].strip()
  return None


def _get_request_origin(scope: dict[str, Any]) -> Optional[str]:
  """Compute the effective origin for the current HTTP/WebSocket request."""
  forwarded = _get_scope_header(scope, b"forwarded")
  if forwarded is not None:
    proto = None
    host = None
    for element in forwarded.split(",", 1)[0].split(";"):
      if "=" not in element:
        continue
      name, value = element.split("=", 1)
      if name.strip().lower() == "proto":
        proto = _strip_optional_quotes(value.strip())
      elif name.strip().lower() == "host":
        host = _strip_optional_quotes(value.strip())
    if proto is not None and host is not None:
      return f"{_normalize_origin_scheme(proto)}://{host}"

  host = _get_scope_header(scope, b"x-forwarded-host")
  if host is None:
    host = _get_scope_header(scope, b"host")
  if host is None:
    return None

  proto = _get_scope_header(scope, b"x-forwarded-proto")
  if proto is None:
    proto = scope.get("scheme", "http")
  return f"{_normalize_origin_scheme(proto)}://{host}"


def _is_request_origin_allowed(
    origin: str,
    scope: dict[str, Any],
    allowed_literal_origins: list[str],
    allowed_origin_regex: Optional[re.Pattern[str]],
    has_configured_allowed_origins: bool,
) -> bool:
  """Validate an Origin header against explicit config or same-origin."""
  if has_configured_allowed_origins and _is_origin_allowed(
      origin, allowed_literal_origins, allowed_origin_regex
  ):
    return True

  request_origin = _get_request_origin(scope)
  if request_origin is None:
    return False
  return origin == request_origin


_SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class _OriginCheckMiddleware:
  """ASGI middleware that blocks cross-origin state-changing requests."""

  def __init__(
      self,
      app: Any,
      has_configured_allowed_origins: bool,
      allowed_origins: list[str],
      allowed_origin_regex: Optional[re.Pattern[str]],
  ) -> None:
    self._app = app
    self._has_configured_allowed_origins = has_configured_allowed_origins
    self._allowed_origins = allowed_origins
    self._allowed_origin_regex = allowed_origin_regex

  async def __call__(
      self,
      scope: dict[str, Any],
      receive: Any,
      send: Any,
  ) -> None:
    if scope["type"] != "http":
      await self._app(scope, receive, send)
      return

    method = scope.get("method", "GET")
    if method in _SAFE_HTTP_METHODS:
      await self._app(scope, receive, send)
      return

    origin = _get_scope_header(scope, b"origin")
    if origin is None:
      await self._app(scope, receive, send)
      return

    if _is_request_origin_allowed(
        origin,
        scope,
        self._allowed_origins,
        self._allowed_origin_regex,
        self._has_configured_allowed_origins,
    ):
      await self._app(scope, receive, send)
      return

    response_body = b"Forbidden: origin not allowed"
    await send({
        "type": "http.response.start",
        "status": 403,
        "headers": [
            (b"content-type", b"text/plain"),
            (b"content-length", str(len(response_body)).encode()),
        ],
    })
    await send({
        "type": "http.response.body",
        "body": response_body,
    })


class ApiServerSpanExporter(export_lib.SpanExporter):

  def __init__(self, trace_dict):
    self.trace_dict = trace_dict

  def export(
      self, spans: typing.Sequence[ReadableSpan]
  ) -> export_lib.SpanExportResult:
    for span in spans:
      if (
          span.name == "call_llm"
          or span.name == "send_data"
          or span.name.startswith("execute_tool")
      ):
        attributes = dict(span.attributes)
        attributes["trace_id"] = span.get_span_context().trace_id
        attributes["span_id"] = span.get_span_context().span_id
        if attributes.get("gcp.vertex.agent.event_id", None):
          self.trace_dict[attributes["gcp.vertex.agent.event_id"]] = attributes
    return export_lib.SpanExportResult.SUCCESS

  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True


class InMemoryExporter(export_lib.SpanExporter):

  def __init__(self, trace_dict):
    super().__init__()
    self._spans = []
    self.trace_dict = trace_dict

  @override
  def export(
      self, spans: typing.Sequence[ReadableSpan]
  ) -> export_lib.SpanExportResult:
    for span in spans:
      trace_id = span.context.trace_id
      if span.name == "call_llm":
        attributes = dict(span.attributes)
        session_id = attributes.get("gcp.vertex.agent.session_id", None)
        if session_id:
          if session_id not in self.trace_dict:
            self.trace_dict[session_id] = [trace_id]
          else:
            self.trace_dict[session_id] += [trace_id]
    self._spans.extend(spans)
    return export_lib.SpanExportResult.SUCCESS

  @override
  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True

  def get_finished_spans(self, session_id: str):
    trace_ids = self.trace_dict.get(session_id, None)
    if trace_ids is None or not trace_ids:
      return []
    return [x for x in self._spans if x.context.trace_id in trace_ids]

  def clear(self):
    self._spans.clear()


class RunAgentRequest(common.BaseModel):
  """Request payload for invoking an agent run."""

  app_name: str = Field(
      description="Application name to run.",
      examples=["hello_world"],
  )
  user_id: str = Field(
      description="User ID whose session is used for the run.",
      examples=["user-123"],
  )
  session_id: str = Field(
      description=(
          "Session ID to run against. The session must already exist unless"
          " the server was configured with auto_create_session=True."
      ),
      examples=["session-abc123"],
  )
  new_message: Optional[types.Content] = Field(
      default=None,
      description=(
          "Optional user message to append before the run starts. Either"
          " new_message or invocation_id is required for a run."
      ),
      examples=[{
          "role": "user",
          "parts": [{"text": "Summarize the last response."}],
      }],
  )
  streaming: bool = Field(
      default=False,
      description=(
          "When true, /run_sse emits server-sent events as they are produced."
          " This flag has no effect on the non-streaming /run endpoint."
      ),
      examples=[True],
  )
  state_delta: Optional[dict[str, Any]] = Field(
      default=None,
      description=(
          "Optional session state changes to apply before the agent run."
      ),
      examples=[{"locale": "en-US"}],
  )
  # for long-running function resume requests (e.g., OAuth callback)
  function_call_event_id: Optional[str] = Field(
      default=None,
      description=(
          "Function call event being resumed. Used by /run_sse to avoid"
          " splitting artifact updates into duplicate content and action"
          " events during resume flows."
      ),
      examples=["event-123"],
  )
  # for resume long-running functions
  invocation_id: Optional[str] = Field(
      default=None,
      description=(
          "Invocation ID to resume. When provided without new_message, the"
          " runner resumes a prior resumable invocation."
      ),
      examples=["invocation-123"],
  )


class CreateSessionRequest(common.BaseModel):
  """Request payload for creating a new session.

  Attributes:
      session_id: The ID of the session to create. If omitted, a random ID is
          generated automatically.
      state: An optional key-value mapping used as the session's initial state.
      events: An optional list of events to seed the session history with.
  """

  session_id: Optional[str] = Field(
      default=None,
      description=(
          "The ID of the session to create. If not provided, a random session"
          " ID will be generated."
      ),
      examples=["session-abc123"],
  )
  state: Optional[dict[str, Any]] = Field(
      default=None,
      description="The initial state of the session.",
      examples=[{"user_preferences": {"language": "en"}}],
  )
  events: Optional[list[Event]] = Field(
      default=None,
      description=(
          "Optional list of events to seed the session history. Events are"
          " appended in order after session creation."
      ),
      examples=[[]],
  )


class SaveArtifactRequest(common.BaseModel):
  """Request payload for saving a new artifact."""

  filename: str = Field(description="Artifact filename.")
  artifact: types.Part = Field(
      description="Artifact payload encoded as google.genai.types.Part."
  )
  custom_metadata: Optional[dict[str, Any]] = Field(
      default=None,
      description="Optional metadata to associate with the artifact version.",
  )


class AddSessionToEvalSetRequest(common.BaseModel):
  eval_id: str
  session_id: str
  user_id: str


class RunEvalRequest(common.BaseModel):
  eval_ids: list[str] = Field(
      deprecated=True,
      default_factory=list,
      description="This field is deprecated, use eval_case_ids instead.",
  )
  eval_case_ids: list[str] = Field(
      default_factory=list,
      description=(
          "List of eval case ids to evaluate. if empty, then all eval cases in"
          " the eval set are run."
      ),
  )
  eval_metrics: list[EvalMetric]


class UpdateMemoryRequest(common.BaseModel):
  """Request to add a session to the memory service."""

  session_id: str
  """The ID of the session to add to memory."""


class UpdateSessionRequest(common.BaseModel):
  """Request to update session state without running the agent."""

  state_delta: dict[str, Any] = Field(
      description="State changes to merge into the existing session state.",
      examples=[{"locale": "en-US"}],
  )
  """The state changes to apply to the session."""


class RunEvalResult(common.BaseModel):
  eval_set_file: str
  eval_set_id: str
  eval_id: str
  final_eval_status: EvalStatus
  eval_metric_results: list[tuple[EvalMetric, EvalMetricResult]] = Field(
      deprecated=True,
      default=[],
      description=(
          "This field is deprecated, use overall_eval_metric_results instead."
      ),
  )
  overall_eval_metric_results: list[EvalMetricResult]
  eval_metric_result_per_invocation: list[EvalMetricResultPerInvocation]
  user_id: str
  session_id: str


class RunEvalResponse(common.BaseModel):
  run_eval_results: list[RunEvalResult]


class GetEventGraphResult(common.BaseModel):
  dot_src: str


class CreateEvalSetRequest(common.BaseModel):
  eval_set: EvalSet


class ListEvalSetsResponse(common.BaseModel):
  eval_set_ids: list[str]


class EvalResult(EvalSetResult):
  """This class has no field intentionally.

  The goal here is to just give a new name to the class to align with the API
  endpoint.
  """


class ListEvalResultsResponse(common.BaseModel):
  eval_result_ids: list[str]


class ListMetricsInfoResponse(common.BaseModel):
  metrics_info: list[MetricInfo]


class AppInfo(common.BaseModel):
  name: str
  root_agent_name: str
  description: str
  language: Literal["yaml", "python"]
  is_computer_use: bool = False


class ListAppsResponse(common.BaseModel):
  apps: list[AppInfo]


def _setup_telemetry(
    otel_to_cloud: bool = False,
    internal_exporters: Optional[list[SpanProcessor]] = None,
):
  # TODO - remove the else branch here once maybe_set_otel_providers is no
  # longer experimental.
  if otel_to_cloud:
    _setup_gcp_telemetry(internal_exporters=internal_exporters)
  elif _otel_env_vars_enabled():
    _setup_telemetry_from_env(internal_exporters=internal_exporters)
  else:
    # Old logic - to be removed when above leaves experimental.
    tracer_provider = TracerProvider()
    if internal_exporters is not None:
      for exporter in internal_exporters:
        tracer_provider.add_span_processor(exporter)
    trace.set_tracer_provider(tracer_provider=tracer_provider)


def _otel_env_vars_enabled() -> bool:
  return any([
      os.getenv(endpoint_var)
      for endpoint_var in [
          otel_env.OTEL_EXPORTER_OTLP_ENDPOINT,
          otel_env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
          otel_env.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
          otel_env.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT,
      ]
  ])


def _setup_gcp_telemetry(
    internal_exporters: list[SpanProcessor] = None,
):
  if typing.TYPE_CHECKING:
    from ..telemetry.setup import OTelHooks

  otel_hooks_to_add: list[OTelHooks] = []

  if internal_exporters:
    from ..telemetry.setup import OTelHooks

    # Register ADK-specific exporters in trace provider.
    otel_hooks_to_add.append(OTelHooks(span_processors=internal_exporters))

  import google.auth

  from ..telemetry.google_cloud import get_gcp_exporters
  from ..telemetry.google_cloud import get_gcp_resource
  from ..telemetry.setup import maybe_set_otel_providers

  credentials, project_id = google.auth.default()

  otel_hooks_to_add.append(
      get_gcp_exporters(
          # TODO - use trace_to_cloud here as well once otel_to_cloud is no
          # longer experimental.
          enable_cloud_tracing=True,
          # TODO - re-enable metrics once errors during shutdown are fixed.
          enable_cloud_metrics=False,
          enable_cloud_logging=True,
          google_auth=(credentials, project_id),
      )
  )
  otel_resource = get_gcp_resource(project_id)

  maybe_set_otel_providers(
      otel_hooks_to_setup=otel_hooks_to_add,
      otel_resource=otel_resource,
  )
  _setup_instrumentation_lib_if_installed()


def _setup_telemetry_from_env(
    internal_exporters: list[SpanProcessor] = None,
):
  from ..telemetry.setup import maybe_set_otel_providers

  otel_hooks_to_add = []

  if internal_exporters:
    from ..telemetry.setup import OTelHooks

    # Register ADK-specific exporters in trace provider.
    otel_hooks_to_add.append(OTelHooks(span_processors=internal_exporters))

  maybe_set_otel_providers(otel_hooks_to_setup=otel_hooks_to_add)
  _setup_instrumentation_lib_if_installed()


def _setup_instrumentation_lib_if_installed():
  # Set instrumentation to enable emitting OTel data from GenAISDK
  # Currently the instrumentation lib is in extras dependencies, make sure to
  # warn the user if it's not installed.
  try:
    from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor

    GoogleGenAiSdkInstrumentor().instrument()
  except ImportError:
    logger.warning(
        "Unable to import GoogleGenAiSdkInstrumentor - some"
        " telemetry will be disabled. Make sure to install google-adk[otel-gcp]"
    )


class AdkWebServer:
  """Helper class for setting up and running the ADK web server on FastAPI.

  You construct this class with all the Services required to run ADK agents and
  can then call the get_fast_api_app method to get a FastAPI app instance that
  can will use your provided service instances, static assets, and agent loader.
  If you pass in a web_assets_dir, the static assets will be served under
  /dev-ui in addition to the API endpoints created by default.

  You can add additional API endpoints by modifying the FastAPI app
  instance returned by get_fast_api_app as this class exposes the agent runners
  and most other bits of state retained during the lifetime of the server.

  Attributes:
      agent_loader: An instance of BaseAgentLoader for loading agents.
      session_service: An instance of BaseSessionService for managing sessions.
      memory_service: An instance of BaseMemoryService for managing memory.
      artifact_service: An instance of BaseArtifactService for managing
        artifacts.
      credential_service: An instance of BaseCredentialService for managing
        credentials.
      eval_sets_manager: An instance of EvalSetsManager for managing evaluation
        sets.
      eval_set_results_manager: An instance of EvalSetResultsManager for
        managing evaluation set results.
      agents_dir: Root directory containing subdirs for agents with those
        containing resources (e.g. .env files, eval sets, etc.) for the agents.
      extra_plugins: A list of fully qualified names of extra plugins to load.
      logo_text: Text to display in the logo of the UI.
      logo_image_url: URL of an image to display as logo of the UI.
      runners_to_clean: Set of runner names marked for cleanup.
      current_app_name_ref: A shared reference to the latest ran app name.
      runner_dict: A dict of instantiated runners for each app.
  """

  def __init__(
      self,
      *,
      agent_loader: BaseAgentLoader,
      session_service: BaseSessionService,
      memory_service: BaseMemoryService,
      artifact_service: BaseArtifactService,
      credential_service: BaseCredentialService,
      eval_sets_manager: EvalSetsManager,
      eval_set_results_manager: EvalSetResultsManager,
      agents_dir: str,
      extra_plugins: Optional[list[str]] = None,
      logo_text: Optional[str] = None,
      logo_image_url: Optional[str] = None,
      url_prefix: Optional[str] = None,
      auto_create_session: bool = False,
  ):
    self.agent_loader = agent_loader
    self.session_service = session_service
    self.memory_service = memory_service
    self.artifact_service = artifact_service
    self.credential_service = credential_service
    self.eval_sets_manager = eval_sets_manager
    self.eval_set_results_manager = eval_set_results_manager
    self.agents_dir = agents_dir
    self.extra_plugins = extra_plugins or []
    self.logo_text = logo_text
    self.logo_image_url = logo_image_url
    # Internal properties we want to allow being modified from callbacks.
    self.runners_to_clean: set[str] = set()
    self.current_app_name_ref: SharedValue[str] = SharedValue(value="")
    self.runner_dict = {}
    self.url_prefix = url_prefix
    self.auto_create_session = auto_create_session

  async def get_runner_async(self, app_name: str) -> Runner:
    """Returns the cached runner for the given app."""
    # Handle cleanup
    if app_name in self.runners_to_clean:
      self.runners_to_clean.remove(app_name)
      runner = self.runner_dict.pop(app_name, None)
      await cleanup.close_runners(list([runner]))

    # Return cached runner if exists
    if app_name in self.runner_dict:
      return self.runner_dict[app_name]

    # Create new runner
    envs.load_dotenv_for_agent(os.path.basename(app_name), self.agents_dir)
    agent_or_app = self.agent_loader.load_agent(app_name)

    # Instantiate extra plugins if configured
    extra_plugins_instances = self._instantiate_extra_plugins()

    if isinstance(agent_or_app, BaseAgent):
      agentic_app = App(
          name=app_name,
          root_agent=agent_or_app,
          plugins=extra_plugins_instances,
      )
    else:
      # Combine existing plugins with extra plugins
      agent_or_app.plugins = agent_or_app.plugins + extra_plugins_instances
      agentic_app = agent_or_app

    runner = self._create_runner(agentic_app)
    self.runner_dict[app_name] = runner
    return runner

  def _get_root_agent(self, agent_or_app: BaseAgent | App) -> BaseAgent:
    """Extract root agent from either a BaseAgent or App object."""
    if isinstance(agent_or_app, App):
      return agent_or_app.root_agent
    return agent_or_app

  def _create_runner(self, agentic_app: App) -> Runner:
    """Create a runner with common services."""
    return Runner(
        app=agentic_app,
        artifact_service=self.artifact_service,
        session_service=self.session_service,
        memory_service=self.memory_service,
        credential_service=self.credential_service,
        auto_create_session=self.auto_create_session,
    )

  def _instantiate_extra_plugins(self) -> list[BasePlugin]:
    """Instantiate extra plugins from the configured list.

    Returns:
      List of instantiated BasePlugin objects.
    """
    extra_plugins_instances = []
    for qualified_name in self.extra_plugins:
      try:
        plugin_obj = self._import_plugin_object(qualified_name)
        if isinstance(plugin_obj, BasePlugin):
          extra_plugins_instances.append(plugin_obj)
        elif issubclass(plugin_obj, BasePlugin):
          extra_plugins_instances.append(plugin_obj(name=qualified_name))
      except Exception as e:
        logger.error("Failed to load plugin %s: %s", qualified_name, e)
    return extra_plugins_instances

  def _import_plugin_object(self, qualified_name: str) -> Any:
    """Import a plugin object (class or instance) from a fully qualified name.

    Args:
      qualified_name: Fully qualified name (e.g.,
        'my_package.my_plugin.MyPlugin')

    Returns:
      The imported object, which can be either a class or an instance.

    Raises:
      ImportError: If the module cannot be imported.
      AttributeError: If the object doesn't exist in the module.
    """
    module_name, obj_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)

  def _setup_runtime_config(self, web_assets_dir: str):
    """Sets up the runtime config for the web server."""
    # Read existing runtime config file.
    runtime_config_path = os.path.join(
        web_assets_dir, "assets", "config", "runtime-config.json"
    )
    runtime_config = {}
    try:
      with open(runtime_config_path, "r") as f:
        runtime_config = json.load(f)
    except FileNotFoundError:
      logger.info(
          "File not found: %s. A new runtime config file will be created.",
          runtime_config_path,
      )
    except json.JSONDecodeError:
      logger.warning(
          "Failed to decode JSON from %s. The file content will be"
          " overwritten.",
          runtime_config_path,
      )
    runtime_config["backendUrl"] = self.url_prefix if self.url_prefix else ""

    # Set custom logo config.
    if self.logo_text or self.logo_image_url:
      if not self.logo_text or not self.logo_image_url:
        raise ValueError(
            "Both --logo-text and --logo-image-url must be defined when using"
            " logo config."
        )
      runtime_config["logo"] = {
          "text": self.logo_text,
          "imageUrl": self.logo_image_url,
      }
    elif "logo" in runtime_config:
      del runtime_config["logo"]

    # Write the runtime config file.
    try:
      os.makedirs(os.path.dirname(runtime_config_path), exist_ok=True)
      with open(runtime_config_path, "w") as f:
        json.dump(runtime_config, f, indent=2)
    except IOError as e:
      logger.error(
          "Failed to write runtime config file %s: %s", runtime_config_path, e
      )

  async def _create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: Optional[str] = None,
      state: Optional[dict[str, Any]] = None,
  ) -> Session:
    try:
      session = await self.session_service.create_session(
          app_name=app_name,
          user_id=user_id,
          state=state,
          session_id=session_id,
      )
      logger.info("New session created: %s", session.id)
      return session
    except AlreadyExistsError as e:
      raise HTTPException(
          status_code=409, detail=f"Session already exists: {session_id}"
      ) from e
    except Exception as e:
      logger.error(
          "Internal server error during session creation: %s", e, exc_info=True
      )
      raise HTTPException(status_code=500, detail=str(e)) from e

  def get_fast_api_app(
      self,
      lifespan: Optional[Lifespan[FastAPI]] = None,
      allow_origins: Optional[list[str]] = None,
      web_assets_dir: Optional[str] = None,
      setup_observer: Callable[
          [Observer, "AdkWebServer"], None
      ] = lambda o, s: None,
      tear_down_observer: Callable[
          [Observer, "AdkWebServer"], None
      ] = lambda o, s: None,
      register_processors: Callable[[TracerProvider], None] = lambda o: None,
      otel_to_cloud: bool = False,
      with_ui: bool = False,
  ):
    """Creates a FastAPI app for the ADK web server.

    By default it'll just return a FastAPI instance with the API server
    endpoints,
    but if you specify a web_assets_dir, it'll also serve the static web assets
    from that directory.

    Args:
      lifespan: The lifespan of the FastAPI app.
      allow_origins: The origins that are allowed to make cross-origin requests.
        Entries can be literal origins (e.g., 'https://example.com') or regex
        patterns prefixed with 'regex:' (e.g.,
        'regex:https://.*\\.example\\.com').
      web_assets_dir: The directory containing the web assets to serve.
      setup_observer: Callback for setting up the file system observer.
      tear_down_observer: Callback for cleaning up the file system observer.
      register_processors: Callback for additional Span processors to be added
        to the TracerProvider.
      otel_to_cloud: Whether to enable Cloud Trace and Cloud Logging
        integrations.

    Returns:
      A FastAPI app instance.
    """
    # Properties we don't need to modify from callbacks
    trace_dict = {}
    session_trace_dict = {}
    # Set up a file system watcher to detect changes in the agents directory.
    observer = Observer()
    setup_observer(observer, self)

    @asynccontextmanager
    async def internal_lifespan(app: FastAPI):
      try:
        if lifespan:
          async with lifespan(app) as lifespan_context:
            yield lifespan_context
        else:
          yield
      finally:
        tear_down_observer(observer, self)
        # Create tasks for all runner closures to run concurrently
        await cleanup.close_runners(list(self.runner_dict.values()))

    memory_exporter = InMemoryExporter(session_trace_dict)

    _setup_telemetry(
        otel_to_cloud=otel_to_cloud,
        internal_exporters=[
            export_lib.SimpleSpanProcessor(ApiServerSpanExporter(trace_dict)),
            export_lib.SimpleSpanProcessor(memory_exporter),
        ],
    )
    if web_assets_dir:
      self._setup_runtime_config(web_assets_dir)

    # TODO - register_processors to be removed once --otel_to_cloud is no
    # longer experimental.
    tracer_provider = trace.get_tracer_provider()
    register_processors(tracer_provider)

    # Run the FastAPI server.
    app = FastAPI(
        title="ADK Web Server API",
        description=(
            "REST API for managing ADK applications, sessions, artifacts, and"
            " agent runs."
        ),
        version=__version__,
        lifespan=internal_lifespan,
    )

    has_configured_allowed_origins = bool(allow_origins)
    if allow_origins:
      literal_origins, combined_regex = _parse_cors_origins(allow_origins)
      compiled_origin_regex = (
          re.compile(combined_regex) if combined_regex is not None else None
      )
      app.add_middleware(
          CORSMiddleware,
          allow_origins=literal_origins,
          allow_origin_regex=combined_regex,
          allow_credentials=True,
          allow_methods=["*"],
          allow_headers=["*"],
      )
    else:
      literal_origins = []
      compiled_origin_regex = None

    app.add_middleware(
        _OriginCheckMiddleware,
        has_configured_allowed_origins=has_configured_allowed_origins,
        allowed_origins=literal_origins,
        allowed_origin_regex=compiled_origin_regex,
    )

    @app.get(
        "/health",
        summary="Health check",
        response_description="Basic liveness check for the API server.",
        responses={
            500: {"description": "Internal server error."},
        },
    )
    async def health() -> dict[str, str]:
      """Check whether the API server is running.

      **Returns**
      - `dict[str, str]`: A small status payload.

      **Errors**
      - `500 Internal Server Error`: Unexpected internal error.
      """
      return {"status": "ok"}

    @app.get(
        "/version",
        summary="Get version information",
        response_description="Version information for ADK and Python runtime.",
        responses={
            500: {"description": "Internal server error."},
        },
    )
    async def version() -> dict[str, str]:
      """Return version information for the server runtime.

      **Returns**
      - `dict[str, str]`: Version metadata containing:
        - `version`: ADK package version.
        - `language`: Always `python`.
        - `language_version`: Python runtime version.

      **Errors**
      - `500 Internal Server Error`: Unexpected internal error.
      """
      return {
          "version": __version__,
          "language": "python",
          "language_version": (
              f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
          ),
      }

    @app.get(
        "/list-apps",
        response_model_exclude_none=True,
        summary="List available apps",
        response_description=(
            "Either a list of app names, or detailed app information when"
            " detailed=true."
        ),
        responses={
            500: {"description": "Internal server error."},
        },
    )
    async def list_apps(
        detailed: bool = Query(
            default=False, description="Return detailed app information"
        )
    ) -> list[str] | ListAppsResponse:
      """List the apps available to the server.

      **Query parameters**
      - `detailed` (optional): When true, returns structured app metadata.

      **Returns**
      - `list[str]`: App names when `detailed` is false.
      - `ListAppsResponse`: App metadata when `detailed` is true.

      **Errors**
      - `500 Internal Server Error`: Unexpected internal error.
      """
      if detailed:
        apps_info = self.agent_loader.list_agents_detailed()
        return ListAppsResponse(apps=[AppInfo(**app) for app in apps_info])
      return self.agent_loader.list_agents()

    @app.get("/debug/trace/{event_id}", tags=[TAG_DEBUG])
    async def get_trace_dict(event_id: str) -> Any:
      event_dict = trace_dict.get(event_id, None)
      if event_dict is None:
        raise HTTPException(status_code=404, detail="Trace not found")
      return event_dict

    if web_assets_dir:

      @app.get("/dev/build_graph/{app_name}")
      async def get_app_info(app_name: str) -> Any:
        runner = await self.get_runner_async(app_name)

        if not runner.app:
          raise HTTPException(
              status_code=404, detail=f"App not found: {app_name}"
          )

        def serialize_agent(agent: BaseAgent) -> dict[str, Any]:
          """Recursively serialize an agent, excluding non-serializable fields."""
          agent_dict = {}

          for field_name, field_info in agent.__class__.model_fields.items():
            # Skip non-serializable fields
            if field_name in [
                "parent_agent",
                "before_agent_callback",
                "after_agent_callback",
                "before_model_callback",
                "after_model_callback",
                "on_model_error_callback",
                "before_tool_callback",
                "after_tool_callback",
                "on_tool_error_callback",
            ]:
              continue

            value = getattr(agent, field_name, None)

            # Handle sub_agents recursively
            if field_name == "sub_agents" and value:
              agent_dict[field_name] = [
                  serialize_agent(sub_agent) for sub_agent in value
              ]
            elif value is None or field_name == "tools":
              continue
            else:
              try:
                if isinstance(value, (str, int, float, bool, list, dict)):
                  agent_dict[field_name] = value
                elif hasattr(value, "model_dump"):
                  agent_dict[field_name] = value.model_dump(
                      mode="python", exclude_none=True
                  )
                else:
                  agent_dict[field_name] = str(value)
              except Exception:
                pass

          return agent_dict

        app_info = {
            "name": runner.app.name,
            "root_agent": serialize_agent(runner.app.root_agent),
        }

        # Add optional fields if present
        if runner.app.plugins:
          app_info["plugins"] = [
              {"name": getattr(plugin, "name", type(plugin).__name__)}
              for plugin in runner.app.plugins
          ]

        if runner.app.context_cache_config:
          try:
            app_info["context_cache_config"] = (
                runner.app.context_cache_config.model_dump(
                    mode="python", exclude_none=True
                )
            )
          except Exception:
            pass

        if runner.app.resumability_config:
          try:
            app_info["resumability_config"] = (
                runner.app.resumability_config.model_dump(
                    mode="python", exclude_none=True
                )
            )
          except Exception:
            pass

        return app_info

    @app.get("/debug/trace/session/{session_id}", tags=[TAG_DEBUG])
    async def get_session_trace(session_id: str) -> Any:
      spans = memory_exporter.get_finished_spans(session_id)
      if not spans:
        return []
      return [
          {
              "name": s.name,
              "span_id": s.context.span_id,
              "trace_id": s.context.trace_id,
              "start_time": s.start_time,
              "end_time": s.end_time,
              "attributes": dict(s.attributes),
              "parent_span_id": s.parent.span_id if s.parent else None,
          }
          for s in spans
      ]

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
        response_model_exclude_none=True,
        summary="Get a session",
        tags=[TAG_SESSIONS],
        response_description=(
            "The stored session, including its current state and event history."
        ),
        responses={
            404: {"description": "Session not found."},
            500: {"description": "Internal server error."},
        },
    )
    async def get_session(
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID to retrieve.",
            examples=["session-abc123"],
        ),
    ) -> Session:
      """Retrieve a specific session by ID.

      Returns the full session object, including conversation history and
      state, for the given application, user, and session ID.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Unique identifier of the session to retrieve.

      **Returns**
      - `Session`: Session object containing conversation events and state.

      **Errors**
      - `404 Not Found`: No session exists with the given ID.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      session = await self.session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      if not session:
        raise HTTPException(status_code=404, detail="Session not found")
      self.current_app_name_ref.value = app_name
      return session

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions",
        response_model_exclude_none=True,
        summary="List sessions",
        tags=[TAG_SESSIONS],
        response_description=(
            "All non-evaluation sessions for the specified application and"
            " user."
        ),
        responses={
            500: {"description": "Internal server error."},
        },
    )
    async def list_sessions(
        app_name: str = Path(
            description="Application name to list sessions from.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID whose sessions are listed.",
            examples=["user-123"],
        ),
    ) -> list[Session]:
      """List sessions for a user in an application.

      Returns all sessions for the given application and user, excluding
      internal sessions generated during evaluation runs.

      **Path parameters**
      - `app_name`: Name of the application.
      - `user_id`: ID of the user whose sessions are listed.

      **Returns**
      - `list[Session]`: List of sessions. Returns an empty list when none
        exist.

      **Errors**
      - `500 Internal Server Error`: Unexpected internal error.
      """
      list_sessions_response = await self.session_service.list_sessions(
          app_name=app_name, user_id=user_id
      )
      return [
          session
          for session in list_sessions_response.sessions
          # Remove sessions that were generated as a part of Eval.
          if not session.id.startswith(EVAL_SESSION_ID_PREFIX)
      ]

    @deprecated(
        "Please use create_session instead. This will be removed in future"
        " releases."
    )
    @app.post(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
        response_model_exclude_none=True,
        summary="Create a session (deprecated)",
        tags=[TAG_SESSIONS],
        response_description="The newly created session.",
        responses={
            409: {"description": "Session already exists."},
            500: {"description": "Internal server error."},
        },
    )
    async def create_session_with_id(
        app_name: str = Path(
            description="Application name under which the session is created.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID for whom the session is created.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID to create.",
            examples=["session-abc123"],
        ),
        state: Optional[dict[str, Any]] = Body(
            default=None,
            description=(
                "Optional initial state for the session. Prefer using"
                " create_session with a request body."
            ),
            examples=[{"locale": "en-US"}],
        ),
    ) -> Session:
      """Create a new session with an explicit session_id (deprecated).

      Prefer `create_session`, which uses a structured request body and can
      seed events. This route exists for backward compatibility and will be
      removed in a future release.

      **Path parameters**
      - `app_name`: Name of the application where the session is created.
      - `user_id`: ID of the user for whom the session is created.
      - `session_id`: Explicit session ID to create.

      **Request body**
      - `state` (optional): Initial session state.

      **Returns**
      - `Session`: Newly created session object.

      **Errors**
      - `409 Conflict`: Session with the same ID already exists.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      return await self._create_session(
          app_name=app_name,
          user_id=user_id,
          state=state,
          session_id=session_id,
      )

    @app.post(
        "/apps/{app_name}/users/{user_id}/sessions",
        response_model_exclude_none=True,
        summary="Create a session",
        tags=[TAG_SESSIONS],
        response_description=(
            "The newly created session with its initial state and any seeded"
            " events."
        ),
        responses={
            409: {"description": "Session already exists."},
            500: {"description": "Internal server error."},
        },
    )
    async def create_session(
        app_name: str = Path(
            description="Application name under which the session is created.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID for whom the session is created.",
            examples=["user-123"],
        ),
        req: Optional[CreateSessionRequest] = None,
    ) -> Session:
      """Create a new session for a user in an application.

      A session stores conversation history and state across agent
      interactions. If `req.session_id` is omitted, a random session ID is
      generated automatically.

      **Path parameters**
      - `app_name`: Name of the application where the session is created.
      - `user_id`: ID of the user for whom the session is created.

      **Request body**
      - `req.session_id` (optional): Explicit session ID.
      - `req.state` (optional): Initial session state.
      - `req.events` (optional): Seed events appended after creation.

      **Returns**
      - `Session`: Newly created session object.

      **Errors**
      - `409 Conflict`: Session with the same ID already exists.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      if not req:
        return await self._create_session(app_name=app_name, user_id=user_id)

      session = await self._create_session(
          app_name=app_name,
          user_id=user_id,
          state=req.state,
          session_id=req.session_id,
      )

      if req.events:
        for event in req.events:
          await self.session_service.append_event(session=session, event=event)

      return session

    @app.delete(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
        summary="Delete a session",
        tags=[TAG_SESSIONS],
        response_description="The session was deleted successfully.",
        responses={
            500: {"description": "Internal server error."},
        },
    )
    async def delete_session(
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID to delete.",
            examples=["session-abc123"],
        ),
    ) -> None:
      """Delete a session for a user in an application.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Unique identifier of the session to delete.

      **Returns**
      - `None`

      **Errors**
      - `500 Internal Server Error`: Unexpected internal error.
      """
      await self.session_service.delete_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )

    @app.patch(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
        response_model_exclude_none=True,
        summary="Update session state",
        tags=[TAG_SESSIONS],
        response_description=(
            "The updated session after applying the provided state delta."
        ),
        responses={
            404: {"description": "Session not found."},
            500: {"description": "Internal server error."},
        },
    )
    async def update_session(
        *,
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID to update.",
            examples=["session-abc123"],
        ),
        req: UpdateSessionRequest,
    ) -> Session:
      """Update session state without running the agent.

      This applies `req.state_delta` by appending a state-delta event to the
      session. The session's persisted `state` is updated by the session
      service based on appended events.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Unique identifier of the session to update.

      **Request body**
      - `req.state_delta`: Key-value state changes to merge into session state.

      **Returns**
      - `Session`: The updated session.

      **Errors**
      - `404 Not Found`: No session exists with the given ID.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      session = await self.session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      if not session:
        raise HTTPException(status_code=404, detail="Session not found")

      # Create an event to record the state change
      import uuid

      from ..events.event import Event
      from ..events.event import EventActions

      state_update_event = Event(
          invocation_id="p-" + str(uuid.uuid4()),
          author="user",
          actions=EventActions(state_delta=req.state_delta),
      )

      # Append the event to the session
      # This will automatically update the session state through __update_session_state
      await self.session_service.append_event(
          session=session, event=state_update_event
      )

      return session

    @app.post(
        "/apps/{app_name}/eval-sets",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def create_eval_set(
        app_name: str, create_eval_set_request: CreateEvalSetRequest
    ) -> EvalSet:
      try:
        return self.eval_sets_manager.create_eval_set(
            app_name=app_name,
            eval_set_id=create_eval_set_request.eval_set.eval_set_id,
        )
      except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve),
        ) from ve

    # TODO - remove after migration
    @deprecated(
        "Please use create_eval_set instead. This will be removed in future"
        " releases."
    )
    @app.post(
        "/apps/{app_name}/eval_sets/{eval_set_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def create_eval_set_legacy(
        app_name: str,
        eval_set_id: str,
    ):
      """Creates an eval set, given the id."""
      await create_eval_set(
          app_name=app_name,
          create_eval_set_request=CreateEvalSetRequest(
              eval_set=EvalSet(eval_set_id=eval_set_id, eval_cases=[])
          ),
      )

    # TODO - remove after migration
    @deprecated(
        "Please use list_eval_sets instead. This will be removed in future"
        " releases."
    )
    @app.get(
        "/apps/{app_name}/eval_sets",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_sets_legacy(app_name: str) -> list[str]:
      list_eval_sets_response = await list_eval_sets(app_name)
      return list_eval_sets_response.eval_set_ids

    # TODO - remove after migration
    @deprecated(
        "Please use run_eval instead. This will be removed in future releases."
    )
    @app.post(
        "/apps/{app_name}/eval_sets/{eval_set_id}/run_eval",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def run_eval_legacy(
        app_name: str, eval_set_id: str, req: RunEvalRequest
    ) -> list[RunEvalResult]:
      run_eval_response = await run_eval(
          app_name=app_name, eval_set_id=eval_set_id, req=req
      )
      return run_eval_response.run_eval_results

    # TODO - remove after migration
    @deprecated(
        "Please use get_eval_result instead. This will be removed in future"
        " releases."
    )
    @app.get(
        "/apps/{app_name}/eval_results/{eval_result_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def get_eval_result_legacy(
        app_name: str,
        eval_result_id: str,
    ) -> EvalSetResult:
      try:
        return self.eval_set_results_manager.get_eval_set_result(
            app_name, eval_result_id
        )
      except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve)) from ve
      except ValidationError as ve:
        raise HTTPException(status_code=500, detail=str(ve)) from ve

    # TODO - remove after migration
    @deprecated(
        "Please use list_eval_results instead. This will be removed in future"
        " releases."
    )
    @app.get(
        "/apps/{app_name}/eval_results",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_results_legacy(app_name: str) -> list[str]:
      list_eval_results_response = await list_eval_results(app_name)
      return list_eval_results_response.eval_result_ids

    @app.get(
        "/apps/{app_name}/eval-sets",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_sets(app_name: str) -> ListEvalSetsResponse:
      """Lists all eval sets for the given app."""
      eval_sets = []
      try:
        eval_sets = self.eval_sets_manager.list_eval_sets(app_name)
      except NotFoundError as e:
        logger.warning(e)

      return ListEvalSetsResponse(eval_set_ids=eval_sets)

    @app.post(
        "/apps/{app_name}/eval-sets/{eval_set_id}/add-session",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    @app.post(
        "/apps/{app_name}/eval_sets/{eval_set_id}/add_session",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def add_session_to_eval_set(
        app_name: str, eval_set_id: str, req: AddSessionToEvalSetRequest
    ):
      # Get the session
      session = await self.session_service.get_session(
          app_name=app_name, user_id=req.user_id, session_id=req.session_id
      )
      assert session, "Session not found."

      # Convert the session data to eval invocations
      invocations = evals.convert_session_to_eval_invocations(session)

      # Populate the session with initial session state.
      agent_or_app = self.agent_loader.load_agent(app_name)
      root_agent = self._get_root_agent(agent_or_app)
      initial_session_state = create_empty_state(root_agent)

      new_eval_case = EvalCase(
          eval_id=req.eval_id,
          conversation=invocations,
          session_input=SessionInput(
              app_name=app_name,
              user_id=req.user_id,
              state=initial_session_state,
          ),
          creation_timestamp=time.time(),
      )

      try:
        self.eval_sets_manager.add_eval_case(
            app_name, eval_set_id, new_eval_case
        )
      except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve

    @app.get(
        "/apps/{app_name}/eval_sets/{eval_set_id}/evals",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_evals_in_eval_set(
        app_name: str,
        eval_set_id: str,
    ) -> list[str]:
      """Lists all evals in an eval set."""
      eval_set_data = self.eval_sets_manager.get_eval_set(app_name, eval_set_id)

      if not eval_set_data:
        raise HTTPException(
            status_code=400, detail=f"Eval set `{eval_set_id}` not found."
        )

      return sorted([x.eval_id for x in eval_set_data.eval_cases])

    @app.get(
        "/apps/{app_name}/eval-sets/{eval_set_id}/eval-cases/{eval_case_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    @app.get(
        "/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def get_eval(
        app_name: str, eval_set_id: str, eval_case_id: str
    ) -> EvalCase:
      """Gets an eval case in an eval set."""
      eval_case_to_find = self.eval_sets_manager.get_eval_case(
          app_name, eval_set_id, eval_case_id
      )

      if eval_case_to_find:
        return eval_case_to_find

      raise HTTPException(
          status_code=404,
          detail=(
              f"Eval set `{eval_set_id}` or Eval `{eval_case_id}` not found."
          ),
      )

    @app.put(
        "/apps/{app_name}/eval-sets/{eval_set_id}/eval-cases/{eval_case_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    @app.put(
        "/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def update_eval(
        app_name: str,
        eval_set_id: str,
        eval_case_id: str,
        updated_eval_case: EvalCase,
    ):
      if (
          updated_eval_case.eval_id
          and updated_eval_case.eval_id != eval_case_id
      ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Eval id in EvalCase should match the eval id in the API route."
            ),
        )

      # Overwrite the value. We are either overwriting the same value or an empty
      # field.
      updated_eval_case.eval_id = eval_case_id
      try:
        self.eval_sets_manager.update_eval_case(
            app_name, eval_set_id, updated_eval_case
        )
      except NotFoundError as nfe:
        raise HTTPException(status_code=404, detail=str(nfe)) from nfe

    @app.delete(
        "/apps/{app_name}/eval-sets/{eval_set_id}/eval-cases/{eval_case_id}",
        tags=[TAG_EVALUATION],
    )
    @app.delete(
        "/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
        tags=[TAG_EVALUATION],
    )
    async def delete_eval(
        app_name: str, eval_set_id: str, eval_case_id: str
    ) -> None:
      try:
        self.eval_sets_manager.delete_eval_case(
            app_name, eval_set_id, eval_case_id
        )
      except NotFoundError as nfe:
        raise HTTPException(status_code=404, detail=str(nfe)) from nfe

    @app.post(
        "/apps/{app_name}/eval-sets/{eval_set_id}/run",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def run_eval(
        app_name: str, eval_set_id: str, req: RunEvalRequest
    ) -> RunEvalResponse:
      """Runs an eval given the details in the eval request."""
      # Create a mapping from eval set file to all the evals that needed to be
      # run.
      try:
        from ..evaluation.local_eval_service import LocalEvalService
        from .cli_eval import _collect_eval_results
        from .cli_eval import _collect_inferences

        eval_set = self.eval_sets_manager.get_eval_set(app_name, eval_set_id)

        if not eval_set:
          raise HTTPException(
              status_code=400, detail=f"Eval set `{eval_set_id}` not found."
          )

        agent_or_app = self.agent_loader.load_agent(app_name)
        root_agent = self._get_root_agent(agent_or_app)

        eval_case_results = []

        eval_service = LocalEvalService(
            root_agent=root_agent,
            eval_sets_manager=self.eval_sets_manager,
            eval_set_results_manager=self.eval_set_results_manager,
            session_service=self.session_service,
            artifact_service=self.artifact_service,
        )
        inference_request = InferenceRequest(
            app_name=app_name,
            eval_set_id=eval_set.eval_set_id,
            eval_case_ids=req.eval_case_ids or req.eval_ids,
            inference_config=InferenceConfig(),
        )
        inference_results = await _collect_inferences(
            inference_requests=[inference_request], eval_service=eval_service
        )

        eval_case_results = await _collect_eval_results(
            inference_results=inference_results,
            eval_service=eval_service,
            eval_metrics=req.eval_metrics,
        )
      except ModuleNotFoundError as e:
        logger.exception("%s", e)
        raise HTTPException(
            status_code=400, detail=MISSING_EVAL_DEPENDENCIES_MESSAGE
        ) from e

      run_eval_results = []
      for eval_case_result in eval_case_results:
        run_eval_results.append(
            RunEvalResult(
                eval_set_file=eval_case_result.eval_set_file,
                eval_set_id=eval_set_id,
                eval_id=eval_case_result.eval_id,
                final_eval_status=eval_case_result.final_eval_status,
                overall_eval_metric_results=eval_case_result.overall_eval_metric_results,
                eval_metric_result_per_invocation=eval_case_result.eval_metric_result_per_invocation,
                user_id=eval_case_result.user_id,
                session_id=eval_case_result.session_id,
            )
        )

      return RunEvalResponse(run_eval_results=run_eval_results)

    @app.get(
        "/apps/{app_name}/eval-results/{eval_result_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def get_eval_result(
        app_name: str,
        eval_result_id: str,
    ) -> EvalResult:
      """Gets the eval result for the given eval id."""
      try:
        eval_set_result = self.eval_set_results_manager.get_eval_set_result(
            app_name, eval_result_id
        )
        return EvalResult(**eval_set_result.model_dump())
      except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve)) from ve
      except ValidationError as ve:
        raise HTTPException(status_code=500, detail=str(ve)) from ve

    @app.get(
        "/apps/{app_name}/eval-results",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_results(app_name: str) -> ListEvalResultsResponse:
      """Lists all eval results for the given app."""
      eval_result_ids = self.eval_set_results_manager.list_eval_set_results(
          app_name
      )
      return ListEvalResultsResponse(eval_result_ids=eval_result_ids)

    @app.get(
        "/apps/{app_name}/metrics-info",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_metrics_info(app_name: str) -> ListMetricsInfoResponse:
      """Lists all eval metrics for the given app."""
      try:
        from ..evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY

        # Right now we ignore the app_name as eval metrics are not tied to the
        # app_name, but they could be moving forward.
        metrics_info = (
            DEFAULT_METRIC_EVALUATOR_REGISTRY.get_registered_metrics()
        )
        return ListMetricsInfoResponse(metrics_info=metrics_info)
      except ModuleNotFoundError as e:
        logger.exception("%s\n%s", MISSING_EVAL_DEPENDENCIES_MESSAGE, e)
        raise HTTPException(
            status_code=400, detail=MISSING_EVAL_DEPENDENCIES_MESSAGE
        ) from e

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
        response_model_exclude_none=True,
        summary="Load an artifact",
        tags=[TAG_ARTIFACTS],
        response_description=(
            "The requested artifact payload for the latest or specified"
            " version."
        ),
        responses={
            404: {"description": "Artifact not found."},
            500: {"description": "Internal server error."},
        },
    )
    async def load_artifact(
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID that owns the artifact.",
            examples=["session-abc123"],
        ),
        artifact_name: str = Path(
            description="Artifact filename to load.",
            examples=["report.txt"],
        ),
        version: Optional[int] = Query(
            default=None,
            description=(
                "Optional artifact version to load. If omitted, the latest"
                " version is returned."
            ),
            examples=[3],
        ),
    ) -> Optional[types.Part]:
      """Load an artifact from a session.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Session ID that owns the artifact.
      - `artifact_name`: Artifact filename to load.

      **Query parameters**
      - `version` (optional): Artifact version to load. Defaults to latest.

      **Returns**
      - `types.Part`: The artifact payload.

      **Errors**
      - `404 Not Found`: Artifact not found.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      artifact = await self.artifact_service.load_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
          version=version,
      )
      if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
      return artifact

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/metadata",
        response_model=list[ArtifactVersion],
        response_model_exclude_none=True,
        summary="List artifact version metadata",
        tags=[TAG_ARTIFACTS],
        response_description=(
            "Metadata for every stored version of the specified artifact."
        ),
        responses={
            500: {"description": "Internal server error."},
        },
    )
    async def list_artifact_versions_metadata(
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID that owns the artifact.",
            examples=["session-abc123"],
        ),
        artifact_name: str = Path(
            description="Artifact filename whose versions are listed.",
            examples=["report.txt"],
        ),
    ) -> list[ArtifactVersion]:
      """List metadata for all stored versions of an artifact.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Session ID that owns the artifact.
      - `artifact_name`: Artifact filename whose versions are listed.

      **Returns**
      - `list[ArtifactVersion]`: Metadata entries for all stored versions.

      **Errors**
      - `500 Internal Server Error`: Unexpected internal error.
      """
      return await self.artifact_service.list_artifact_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
      )

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/{version_id}",
        response_model_exclude_none=True,
        summary="Load a specific artifact version",
        tags=[TAG_ARTIFACTS],
        response_description="The artifact payload for the requested version.",
        responses={
            404: {"description": "Artifact not found."},
            500: {"description": "Internal server error."},
        },
    )
    async def load_artifact_version(
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID that owns the artifact.",
            examples=["session-abc123"],
        ),
        artifact_name: str = Path(
            description="Artifact filename to load.",
            examples=["report.txt"],
        ),
        version_id: int = Path(
            description="Artifact version number to load.",
            examples=[3],
        ),
    ) -> Optional[types.Part]:
      """Load a specific stored version of an artifact.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Session ID that owns the artifact.
      - `artifact_name`: Artifact filename to load.
      - `version_id`: Artifact version number to load.

      **Returns**
      - `types.Part`: The artifact payload.

      **Errors**
      - `404 Not Found`: Artifact not found.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      artifact = await self.artifact_service.load_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
          version=version_id,
      )
      if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
      return artifact

    @app.post(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts",
        response_model=ArtifactVersion,
        response_model_exclude_none=True,
        summary="Save an artifact",
        tags=[TAG_ARTIFACTS],
        response_description="Metadata for the newly created artifact version.",
        responses={
            400: {"description": "Invalid artifact payload."},
            500: {"description": "Internal server error."},
        },
    )
    async def save_artifact(
        *,
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID that will own the saved artifact.",
            examples=["session-abc123"],
        ),
        req: SaveArtifactRequest,
    ) -> ArtifactVersion:
      """Save a new artifact version in a session.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Session ID that will own the saved artifact.

      **Request body**
      - `req.filename`: Artifact filename.
      - `req.artifact`: Artifact payload as `google.genai.types.Part`.
      - `req.custom_metadata` (optional): Metadata attached to the version.

      **Returns**
      - `ArtifactVersion`: Metadata for the newly created artifact version.

      **Errors**
      - `400 Bad Request`: Invalid artifact payload.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      try:
        version = await self.artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=req.filename,
            artifact=req.artifact,
            custom_metadata=req.custom_metadata,
        )
      except InputValidationError as ive:
        raise HTTPException(status_code=400, detail=str(ive)) from ive
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error(
            "Internal error while saving artifact %s for app=%s user=%s"
            " session=%s: %s",
            req.filename,
            app_name,
            user_id,
            session_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
      artifact_version = await self.artifact_service.get_artifact_version(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=req.filename,
          version=version,
      )
      if artifact_version is None:
        raise HTTPException(
            status_code=500, detail="Artifact metadata unavailable"
        )
      return artifact_version

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/{version_id}/metadata",
        response_model=ArtifactVersion,
        response_model_exclude_none=True,
        summary="Get artifact version metadata",
        tags=[TAG_ARTIFACTS],
        response_description="Metadata for the requested artifact version.",
        responses={
            404: {"description": "Artifact version not found."},
            500: {"description": "Internal server error."},
        },
    )
    async def get_artifact_version_metadata(
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID that owns the artifact.",
            examples=["session-abc123"],
        ),
        artifact_name: str = Path(
            description="Artifact filename whose metadata is retrieved.",
            examples=["report.txt"],
        ),
        version_id: int = Path(
            description="Artifact version number whose metadata is retrieved.",
            examples=[3],
        ),
    ) -> ArtifactVersion:
      """Get metadata for a specific artifact version.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Session ID that owns the artifact.
      - `artifact_name`: Artifact filename.
      - `version_id`: Artifact version number.

      **Returns**
      - `ArtifactVersion`: Metadata for the requested artifact version.

      **Errors**
      - `404 Not Found`: Artifact version not found.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      artifact_version = await self.artifact_service.get_artifact_version(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
          version=version_id,
      )
      if not artifact_version:
        raise HTTPException(
            status_code=404, detail="Artifact version not found"
        )
      return artifact_version

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts",
        response_model_exclude_none=True,
        summary="List artifact names",
        tags=[TAG_ARTIFACTS],
        response_description="All artifact names stored for the session.",
        responses={
            500: {"description": "Internal server error."},
        },
    )
    async def list_artifact_names(
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID whose artifact names are listed.",
            examples=["session-abc123"],
        ),
    ) -> list[str]:
      """List all artifact names stored in a session.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Session ID whose artifact names are listed.

      **Returns**
      - `list[str]`: Artifact filenames stored for the session.

      **Errors**
      - `500 Internal Server Error`: Unexpected internal error.
      """
      return await self.artifact_service.list_artifact_keys(
          app_name=app_name, user_id=user_id, session_id=session_id
      )

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions",
        response_model_exclude_none=True,
        summary="List artifact versions",
        tags=[TAG_ARTIFACTS],
        response_description=(
            "Version numbers available for the specified artifact."
        ),
        responses={
            500: {"description": "Internal server error."},
        },
    )
    async def list_artifact_versions(
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID that owns the artifact.",
            examples=["session-abc123"],
        ),
        artifact_name: str = Path(
            description="Artifact filename whose versions are listed.",
            examples=["report.txt"],
        ),
    ) -> list[int]:
      """List all stored version numbers for an artifact.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Session ID that owns the artifact.
      - `artifact_name`: Artifact filename whose versions are listed.

      **Returns**
      - `list[int]`: Version numbers available for the artifact.

      **Errors**
      - `500 Internal Server Error`: Unexpected internal error.
      """
      return await self.artifact_service.list_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
      )

    @app.delete(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
        summary="Delete an artifact",
        tags=[TAG_ARTIFACTS],
        response_description="The artifact was deleted successfully.",
        responses={
            500: {"description": "Internal server error."},
        },
    )
    async def delete_artifact(
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID that owns the artifact.",
            examples=["session-abc123"],
        ),
        artifact_name: str = Path(
            description="Artifact filename to delete.",
            examples=["report.txt"],
        ),
    ) -> None:
      """Delete an artifact and all its stored versions from a session.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Session ID that owns the artifact.
      - `artifact_name`: Artifact filename to delete.

      **Returns**
      - `None`

      **Errors**
      - `500 Internal Server Error`: Unexpected internal error.
      """
      await self.artifact_service.delete_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
      )

    @app.patch(
        "/apps/{app_name}/users/{user_id}/memory",
        summary="Add a session to memory",
        tags=[TAG_MEMORY],
        response_description=(
            "The session events were added to the configured memory service."
        ),
        responses={
            400: {
                "description": (
                    "Memory service is not configured or request is invalid."
                )
            },
            404: {"description": "Session not found."},
            500: {"description": "Internal server error."},
        },
    )
    async def patch_memory(
        *,
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        update_memory_request: UpdateMemoryRequest,
    ) -> None:
      """Add all events from a session to the configured memory service.

      **Path parameters**
      - `app_name`: Name of the application.
      - `user_id`: ID of the user.

      **Request body**
      - `update_memory_request.session_id`: Session ID whose events are added
        to memory.

      **Returns**
      - `None`

      **Errors**
      - `400 Bad Request`: Memory service not configured or request invalid.
      - `404 Not Found`: Session not found.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      if not self.memory_service:
        raise HTTPException(
            status_code=400, detail="Memory service is not configured."
        )
      if (
          update_memory_request is None
          or update_memory_request.session_id is None
      ):
        raise HTTPException(
            status_code=400, detail="Update memory request is invalid."
        )

      session = await self.session_service.get_session(
          app_name=app_name,
          user_id=user_id,
          session_id=update_memory_request.session_id,
      )
      if not session:
        raise HTTPException(status_code=404, detail="Session not found")
      await self.memory_service.add_session_to_memory(session)

    @app.post(
        "/run",
        response_model_exclude_none=True,
        summary="Run an agent",
        tags=[TAG_RUNS],
        response_description=(
            "Ordered events generated while processing the request."
        ),
        responses={
            404: {"description": "Session not found."},
            422: {"description": "Invalid request payload."},
            500: {"description": "Internal server error."},
        },
    )
    async def run_agent(req: RunAgentRequest) -> list[Event]:
      """Run an agent against a session and return all generated events.

      This endpoint runs the app identified by `req.app_name` using the
      provided session identifiers. Events are returned as a list after the
      invocation finishes.

      **Request body**
      - `req.app_name`: Application name to run.
      - `req.user_id`: User ID that owns the session.
      - `req.session_id`: Session ID to run against.
      - `req.new_message` (optional): New user message to append before
        running. If provided without a role, it is treated as a user message.
      - `req.state_delta` (optional): Session state changes to apply before
        running.
      - `req.invocation_id` (optional): Invocation ID to resume (resumable
        apps only). Either `new_message` or `invocation_id` must be provided.

      **Returns**
      - `list[Event]`: Ordered events generated for the invocation.

      **Errors**
      - `404 Not Found`: Session not found when the runner is not configured
        to auto-create sessions.
      - `422 Unprocessable Entity`: Invalid request payload.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      runner = await self.get_runner_async(req.app_name)
      try:
        async with Aclosing(
            runner.run_async(
                user_id=req.user_id,
                session_id=req.session_id,
                new_message=req.new_message,
                state_delta=req.state_delta,
                invocation_id=req.invocation_id,
            )
        ) as agen:
          events = [event async for event in agen]
      except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
      logger.info("Generated %s events in agent run", len(events))
      logger.debug("Events generated: %s", events)
      return events

    @app.post(
        "/run_sse",
        summary="Run an agent with server-sent events",
        tags=[TAG_RUNS],
        response_description=(
            "A text/event-stream response that emits events as they are"
            " generated."
        ),
        responses={
            404: {"description": "Session not found."},
            422: {"description": "Invalid request payload."},
            500: {"description": "Internal server error."},
        },
    )
    async def run_agent_sse(req: RunAgentRequest) -> StreamingResponse:
      """Run an agent and stream generated events using server-sent events.

      This endpoint streams events as they are generated using the
      `text/event-stream` media type. Each SSE message is formatted as:
      `data: {"key":"value"}`, where the JSON object is an `Event`
      serialized with camelCase field names.

      **Request body**
      - `req.app_name`: Application name to run.
      - `req.user_id`: User ID that owns the session.
      - `req.session_id`: Session ID to run against.
      - `req.new_message` (optional): New user message to append before
        running.
      - `req.state_delta` (optional): Session state changes to apply before
        running.
      - `req.invocation_id` (optional): Invocation ID to resume (resumable
        apps only).
      - `req.streaming` (optional): Controls the runner streaming mode. The
        response itself is always SSE.
      - `req.function_call_event_id` (optional): When omitted, some artifact
        updates may be split into two SSE events (content-only and action-only)
        to match the ADK Web UI rendering behavior.

      **Returns**
      - `StreamingResponse`: SSE stream of `Event` payloads.

      **Errors**
      - `404 Not Found`: Session not found when the runner is not configured
        to auto-create sessions.
      - `422 Unprocessable Entity`: Invalid request payload.
      - `500 Internal Server Error`: Unexpected internal error.
      """
      stream_mode = StreamingMode.SSE if req.streaming else StreamingMode.NONE
      runner = await self.get_runner_async(req.app_name)

      # Validate session existence before starting the stream.
      # We check directly here instead of eagerly advancing the
      # runner's async generator with anext(), because splitting
      # generator consumption across two asyncio Tasks (request
      # handler vs StreamingResponse) breaks OpenTelemetry context
      # detachment.
      if not runner.auto_create_session:
        session = await self.session_service.get_session(
            app_name=req.app_name,
            user_id=req.user_id,
            session_id=req.session_id,
        )
        if not session:
          raise HTTPException(
              status_code=404,
              detail=f"Session not found: {req.session_id}",
          )

      # Convert the events to properly formatted SSE
      async def event_generator():
        async with Aclosing(
            runner.run_async(
                user_id=req.user_id,
                session_id=req.session_id,
                new_message=req.new_message,
                state_delta=req.state_delta,
                run_config=RunConfig(streaming_mode=stream_mode),
                invocation_id=req.invocation_id,
            )
        ) as agen:
          try:
            async for event in agen:
              # ADK Web renders artifacts from `actions.artifactDelta`
              # during part processing *and* during action processing
              # 1) the original event with `artifactDelta` cleared (content)
              # 2) a content-less "action-only" event carrying `artifactDelta`
              events_to_stream = [event]
              if (
                  not req.function_call_event_id
                  and event.actions.artifact_delta
                  and event.content
                  and event.content.parts
              ):
                content_event = event.model_copy(deep=True)
                content_event.actions.artifact_delta = {}
                artifact_event = event.model_copy(deep=True)
                artifact_event.content = None
                events_to_stream = [content_event, artifact_event]

              for event_to_stream in events_to_stream:
                sse_event = event_to_stream.model_dump_json(
                    exclude_none=True,
                    by_alias=True,
                )
                logger.debug(
                    "Generated event in agent run streaming: %s", sse_event
                )
                yield f"data: {sse_event}\n\n"
          except Exception as e:
            logger.exception("Error in event_generator: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

      # Returns a streaming response with the proper media type for SSE
      return StreamingResponse(
          event_generator(),
          media_type="text/event-stream",
      )

    @app.get(
        "/dev/{app_name}/graph",
        response_model_exclude_none=True,
        tags=[TAG_DEBUG],
    )
    async def get_app_graph_dot(
        app_name: str, dark_mode: bool = False
    ) -> GetEventGraphResult | dict:
      """Returns the base agent graph in DOT format without any highlights.

      This endpoint allows the frontend to fetch the graph structure once
      and compute highlights client-side for better performance.

      Args:
        app_name: The name of the agent/app
        dark_mode: Whether to use dark theme background color
      """
      agent_or_app = self.agent_loader.load_agent(app_name)
      root_agent = self._get_root_agent(agent_or_app)

      # Get graph with NO highlights (empty list) and specified theme
      dot_graph = await agent_graph.get_agent_graph(
          root_agent, [], dark_mode=dark_mode
      )

      if dot_graph and isinstance(dot_graph, graphviz.Digraph):
        return GetEventGraphResult(dot_src=dot_graph.source)
      else:
        return {}

    # TODO: This endpoint can be removed once we update adk web to stop consuming it
    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/events/{event_id}/graph",
        response_model_exclude_none=True,
        summary="Get an event graph (debug)",
        response_description=(
            "Graphviz DOT source highlighting the agent/tool edge(s) inferred"
            " from the specified event. Returns an empty object when the event"
            " is not found."
        ),
        tags=[TAG_DEBUG],
        responses={
            500: {"description": "Internal server error."},
        },
    )
    async def get_event_graph(
        app_name: str = Path(
            description="Application name that owns the session.",
            examples=["hello_world"],
        ),
        user_id: str = Path(
            description="User ID that owns the session.",
            examples=["user-123"],
        ),
        session_id: str = Path(
            description="Session ID that owns the event.",
            examples=["session-abc123"],
        ),
        event_id: str = Path(
            description="Event ID to visualize.",
            examples=["event-123"],
        ),
    ):
      """Get a Graphviz DOT graph for a session event (debug endpoint).

      This endpoint inspects the specified event and highlights agent-to-tool
      (function call) and tool-to-agent (function response) edges in the agent
      graph. If the event cannot be found, an empty object is returned.

      **Path parameters**
      - `app_name`: Name of the application that owns the session.
      - `user_id`: ID of the user who owns the session.
      - `session_id`: Session ID that owns the event.
      - `event_id`: Event ID to visualize.

      **Returns**
      - `GetEventGraphResult`: Contains `dot_src` (Graphviz DOT source).

      **Errors**
      - `500 Internal Server Error`: Unexpected internal error.
      """
      session = await self.session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      session_events = session.events if session else []
      event = next((x for x in session_events if x.id == event_id), None)
      if not event:
        return {}

      function_calls = event.get_function_calls()
      function_responses = event.get_function_responses()
      agent_or_app = self.agent_loader.load_agent(app_name)
      root_agent = self._get_root_agent(agent_or_app)
      dot_graph = None
      if function_calls:
        function_call_highlights = []
        for function_call in function_calls:
          from_name = event.author
          to_name = function_call.name
          function_call_highlights.append((from_name, to_name))
          dot_graph = await agent_graph.get_agent_graph(
              root_agent, function_call_highlights
          )
      elif function_responses:
        function_responses_highlights = []
        for function_response in function_responses:
          from_name = function_response.name
          to_name = event.author
          function_responses_highlights.append((from_name, to_name))
          dot_graph = await agent_graph.get_agent_graph(
              root_agent, function_responses_highlights
          )
      else:
        from_name = event.author
        to_name = ""
        dot_graph = await agent_graph.get_agent_graph(
            root_agent, [(from_name, to_name)]
        )
      if dot_graph and isinstance(dot_graph, graphviz.Digraph):
        return GetEventGraphResult(dot_src=dot_graph.source)
      else:
        return {}

    @app.websocket("/run_live")
    async def run_agent_live(
        websocket: WebSocket,
        app_name: str,
        user_id: str,
        session_id: str,
        modalities: List[Literal["TEXT", "AUDIO"]] = Query(
            default=["AUDIO"]
        ),  # Only allows "TEXT" or "AUDIO"
        proactive_audio: bool | None = Query(default=None),
        enable_affective_dialog: bool | None = Query(default=None),
        enable_session_resumption: bool | None = Query(default=None),
    ) -> None:
      ws_origin = websocket.headers.get("origin")
      if ws_origin is not None and not _is_request_origin_allowed(
          ws_origin,
          websocket.scope,
          literal_origins,
          compiled_origin_regex,
          has_configured_allowed_origins,
      ):
        await websocket.close(code=1008, reason="Origin not allowed")
        return

      await websocket.accept()

      session = await self.session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      if not session:
        await websocket.close(code=1002, reason="Session not found")
        return

      live_request_queue = LiveRequestQueue()

      async def forward_events():
        runner = await self.get_runner_async(app_name)
        run_config = RunConfig(
            response_modalities=modalities,
            proactivity=(
                types.ProactivityConfig(proactive_audio=proactive_audio)
                if proactive_audio is not None
                else None
            ),
            enable_affective_dialog=enable_affective_dialog,
            session_resumption=(
                types.SessionResumptionConfig(
                    transparent=enable_session_resumption
                )
                if enable_session_resumption is not None
                else None
            ),
        )
        async with Aclosing(
            runner.run_live(
                session=session,
                live_request_queue=live_request_queue,
                run_config=run_config,
            )
        ) as agen:
          async for event in agen:
            await websocket.send_text(
                event.model_dump_json(exclude_none=True, by_alias=True)
            )

      async def process_messages():
        try:
          while True:
            data = await websocket.receive_text()
            # Validate and send the received message to the live queue.
            live_request_queue.send(LiveRequest.model_validate_json(data))
        except ValidationError as ve:
          logger.error("Validation error in process_messages: %s", ve)

      # Run both tasks concurrently and cancel all if one fails.
      tasks = [
          asyncio.create_task(forward_events()),
          asyncio.create_task(process_messages()),
      ]
      done, pending = await asyncio.wait(
          tasks, return_when=asyncio.FIRST_EXCEPTION
      )
      try:
        # This will re-raise any exception from the completed tasks.
        for task in done:
          task.result()
      except WebSocketDisconnect:
        # Disconnection could happen when receive or send text via websocket
        logger.info("Client disconnected during live session.")
      except Exception as e:
        logger.exception("Error during live websocket communication: %s", e)
        traceback.print_exc()
        WEBSOCKET_INTERNAL_ERROR_CODE = 1011
        WEBSOCKET_MAX_BYTES_FOR_REASON = 123
        await websocket.close(
            code=WEBSOCKET_INTERNAL_ERROR_CODE,
            reason=str(e)[:WEBSOCKET_MAX_BYTES_FOR_REASON],
        )
      finally:
        for task in pending:
          task.cancel()

    if web_assets_dir:
      import mimetypes

      mimetypes.add_type("application/javascript", ".js", True)
      mimetypes.add_type("text/javascript", ".js", True)

      redirect_dev_ui_url = (
          self.url_prefix + "/dev-ui/" if self.url_prefix else "/dev-ui/"
      )

      @app.get("/dev-ui/config")
      async def get_ui_config():
        return {
            "logo_text": self.logo_text,
            "logo_image_url": self.logo_image_url,
        }

      @app.get("/")
      async def redirect_root_to_dev_ui():
        return RedirectResponse(redirect_dev_ui_url)

      @app.get("/dev-ui")
      async def redirect_dev_ui_add_slash():
        return RedirectResponse(redirect_dev_ui_url)

      app.mount(
          "/dev-ui/",
          StaticFiles(directory=web_assets_dir, html=True, follow_symlink=True),
          name="static",
      )

    return app
