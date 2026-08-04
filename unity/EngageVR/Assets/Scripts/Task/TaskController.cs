// EngageVR -- the desktop reaction task MonoBehaviour.
//
// DESKTOP ONLY. Monitor, keyboard, mouse. No headset, no OpenXR, no XR
// Interaction Toolkit, no VR hardware, and no dependency on any of
// them.
//
// SCOPE. This is a software telemetry source. Its accuracy, reaction
// times, and timeouts are not engagement, attention, cognitive-load, or
// fatigue measurements. The task has not been experimentally designed,
// piloted, or approved.
//
// The controller owns the loop; the trial logic, telemetry building,
// and command handling live in plain C# classes that the EditMode tests
// drive without entering play mode.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace EngageVR.Task
{
    using EngageVR.Networking;
    using EngageVR.Protocol;

    /// <summary>Runs the desktop reaction task and speaks the protocol.</summary>
    public sealed class TaskController : MonoBehaviour
    {
        [Header("Connection")]
        [Tooltip("Leave 'Use Mock Transport' on to run with no server at all.")]
        public bool useMockTransport = true;

        [Tooltip("Backend WebSocket URL. Loopback only; there is no authentication.")]
        public string webSocketUrl = "ws://127.0.0.1:8000/ws/v1/sessions/unity-session";

        public string sessionId = "unity-session";
        public int maximumMessageBytes = 262144;

        [Header("Task")]
        public string taskId = "reaction_task_v1";

        [Tooltip("Pseudonymous label only. Never a real name or identifier.")]
        public string participantId = "unity_desktop_user";

        public int blocks = 2;
        public int trialsPerBlock = 10;
        public int seed = 42;
        public int defaultDifficulty = 1;
        public float interTrialIntervalMs = 500f;
        public float responseTimeoutMs = 1500f;
        public float feedbackDurationMs = 400f;

        [Header("Input")]
        public KeyCode[] responseKeys = { KeyCode.J, KeyCode.K, KeyCode.L };

        public KeyCode startKey = KeyCode.Space;

        /// <summary>Read by the HUD. Never mutated by it.</summary>
        public TaskRuntimeState RuntimeState { get; private set; }

        public IMessageTransport Transport { get; private set; }

        public TaskTelemetry Telemetry { get; private set; }

        public string LastFeedback { get; private set; }

        public string CurrentStimulusCategory { get; private set; }

        public bool StimulusVisible
        {
            get { return _trial != null && _trial.StimulusVisible; }
        }

        public int CompletedTrialCount { get; private set; }

        public int TimeoutCount { get; private set; }

        public string ConnectionDetail
        {
            get { return Transport == null ? "no transport" : Transport.StatusDetail; }
        }

        private AdaptationReceiver _adaptation;
        private TrialController _trial;
        private List<TrialDefinition> _plan;
        private int _planIndex;
        private double _taskStartMs;
        private double _nowMs;
        private double _nextTrialAtMs;
        private double _feedbackUntilMs;
        private double _lastHeartbeatMs;
        private double _heartbeatIntervalMs = 10000.0;
        private int _heartbeatCounter;
        private bool _sessionStarted;
        private bool _sessionEnded;

        private void Awake()
        {
            RuntimeState = new TaskRuntimeState
            {
                State = TaskStateName.Idle,
                DifficultyLevel = defaultDifficulty,
                StimulusIntervalMs = interTrialIntervalMs
            };
            _adaptation = new AdaptationReceiver(RuntimeState);
            _trial = new TrialController(responseTimeoutMs);
            _plan = TrialPlanBuilder.Build(blocks, trialsPerBlock, seed);
        }

        private void Start()
        {
            Transport = useMockTransport
                ? (IMessageTransport)new MockTransport()
                : new WebSocketTransport(webSocketUrl, maximumMessageBytes);
            Transport.MessageReceived += OnMessageReceived;
            Transport.Faulted += OnTransportFaulted;

            // The task client is a live participant in the conversation.
            // It reports itself as live because a human is pressing the
            // keys; the Python simulator, whose responses are fabricated,
            // reports itself as synthetic instead.
            EnvelopeFactory factory = new EnvelopeFactory(
                sessionId,
                MessageSource.UnityClient,
                MessageProvenance.Live("engagevr.unity.task_controller"),
                () => _nowMs / 1000.0);
            Telemetry = new TaskTelemetry(factory, taskId);

            Transport.Connect();
            Transport.Send(Telemetry.ClientHello("engagevr-unity-client", "0.1.0"));
        }

        private void Update()
        {
            _nowMs += Time.unscaledDeltaTime * 1000.0;
            Transport.Poll();
            SendHeartbeatIfDue();

            if (!_sessionStarted)
            {
                if (Input.GetKeyDown(startKey))
                {
                    BeginSession();
                }

                return;
            }

            if (_sessionEnded
                || string.Equals(RuntimeState.State, TaskStateName.Paused, StringComparison.Ordinal))
            {
                return;
            }

            if (_trial.Resolution == TrialResolution.Pending && _trial.Current != null)
            {
                PollResponseKeys();
                if (_trial.Tick(_nowMs))
                {
                    ResolveTimeout();
                }

                return;
            }

            if (_nowMs >= _nextTrialAtMs)
            {
                AdvanceToNextTrial();
            }
        }

        private void OnDestroy()
        {
            if (Transport == null)
            {
                return;
            }

            if (_sessionStarted && !_sessionEnded)
            {
                // A task that is torn down mid-run still reports an end,
                // marked incomplete. A truncated recording must not be
                // indistinguishable from a finished one.
                EmitTaskEvent(
                    Telemetry.Lifecycle(
                        TaskEventType.TaskAborted, TaskElapsedMs, RuntimeState.DifficultyLevel));
                Transport.Send(Telemetry.SessionEnd(false, "client_destroyed"));
            }

            Transport.MessageReceived -= OnMessageReceived;
            Transport.Faulted -= OnTransportFaulted;
            Transport.Dispose();
        }

        private double TaskElapsedMs
        {
            get { return Math.Max(_nowMs - _taskStartMs, 0.0); }
        }

        // -- session lifecycle ---------------------------------------------

        private void BeginSession()
        {
            _sessionStarted = true;
            _taskStartMs = _nowMs;
            _planIndex = 0;
            CompletedTrialCount = 0;
            TimeoutCount = 0;

            Dictionary<string, JsonValue> configuration = new Dictionary<string, JsonValue>
            {
                { "seed", JsonValue.Number(seed) },
                { "response_timeout_ms", JsonValue.Number(responseTimeoutMs) },
                { "inter_trial_interval_ms", JsonValue.Number(interTrialIntervalMs) },
                { "client", JsonValue.String("unity_desktop") }
            };
            Transport.Send(
                Telemetry.SessionStart(
                    participantId,
                    blocks,
                    trialsPerBlock,
                    RuntimeState.DifficultyLevel,
                    configuration));

            RuntimeState.State = TaskStateName.Loaded;
            EmitTaskEvent(
                Telemetry.Lifecycle(TaskEventType.TaskLoaded, 0.0, RuntimeState.DifficultyLevel));
            RuntimeState.State = TaskStateName.Running;
            EmitTaskEvent(
                Telemetry.Lifecycle(TaskEventType.TaskStarted, 0.0, RuntimeState.DifficultyLevel));
            EmitTaskState();

            EmitTaskEvent(
                Telemetry.BlockEvent(
                    TaskEventType.BlockStarted, 0, TaskElapsedMs, RuntimeState.DifficultyLevel));
            _nextTrialAtMs = _nowMs;
        }

        private void CompleteSession()
        {
            _sessionEnded = true;
            RuntimeState.State = TaskStateName.Completed;
            EmitTaskEvent(
                Telemetry.Lifecycle(
                    TaskEventType.TaskCompleted, TaskElapsedMs, RuntimeState.DifficultyLevel));
            EmitTaskState();
            Transport.Send(Telemetry.SessionEnd(true, "task_completed"));
            LastFeedback = "Task complete.";
        }

        // -- trial loop -----------------------------------------------------

        private void AdvanceToNextTrial()
        {
            if (_planIndex >= _plan.Count)
            {
                EmitTaskEvent(
                    Telemetry.BlockEvent(
                        TaskEventType.BlockCompleted,
                        _plan[_plan.Count - 1].BlockId,
                        TaskElapsedMs,
                        RuntimeState.DifficultyLevel));
                CompleteSession();
                return;
            }

            TrialDefinition definition = _plan[_planIndex];
            if (_planIndex > 0 && definition.BlockId != _plan[_planIndex - 1].BlockId)
            {
                EmitTaskEvent(
                    Telemetry.BlockEvent(
                        TaskEventType.BlockCompleted,
                        _plan[_planIndex - 1].BlockId,
                        TaskElapsedMs,
                        RuntimeState.DifficultyLevel));
                EmitTaskEvent(
                    Telemetry.BlockEvent(
                        TaskEventType.BlockStarted,
                        definition.BlockId,
                        TaskElapsedMs,
                        RuntimeState.DifficultyLevel));
            }

            RuntimeState.BlockId = definition.BlockId;
            RuntimeState.TrialId = definition.TrialId;
            _planIndex++;

            EmitTaskEvent(
                Telemetry.TrialStarted(
                    definition.BlockId,
                    definition.TrialId,
                    TaskElapsedMs,
                    RuntimeState.DifficultyLevel));

            _trial.Begin(definition, _nowMs);
            CurrentStimulusCategory = definition.StimulusCategory;
            LastFeedback = string.Empty;

            EmitTaskEvent(
                Telemetry.StimulusPresented(
                    definition.BlockId,
                    definition.TrialId,
                    definition.StimulusId,
                    definition.StimulusCategory,
                    definition.ExpectedResponse,
                    TaskElapsedMs,
                    0.0,
                    RuntimeState.DifficultyLevel));
        }

        private void PollResponseKeys()
        {
            foreach (KeyCode key in responseKeys)
            {
                if (!Input.GetKeyDown(key))
                {
                    continue;
                }

                string pressed = key.ToString().ToLowerInvariant();
                if (_trial.RegisterResponse(pressed, _nowMs))
                {
                    ResolveResponse();
                }

                return;
            }
        }

        private void ResolveResponse()
        {
            TrialDefinition definition = _trial.Current;
            bool correct = _trial.ResponseCorrect.HasValue && _trial.ResponseCorrect.Value;

            TaskEventDetail detail = Telemetry.ResponseRegistered(
                definition.BlockId,
                definition.TrialId,
                definition.StimulusId,
                definition.StimulusCategory,
                definition.ExpectedResponse,
                _trial.ObservedResponse,
                correct,
                _trial.ReactionTimeMs.Value,
                TaskElapsedMs,
                _trial.ElapsedSinceStimulusMs(_nowMs),
                RuntimeState.DifficultyLevel);
            EmitTaskEvent(detail);
            EmitTaskEvent(Telemetry.TrialCompleted(detail));

            CompletedTrialCount++;
            CurrentStimulusCategory = null;
            LastFeedback = correct ? "correct" : "incorrect";
            ScheduleNextTrial();
        }

        private void ResolveTimeout()
        {
            TrialDefinition definition = _trial.Current;
            TaskEventDetail detail = Telemetry.ResponseTimeout(
                definition.BlockId,
                definition.TrialId,
                definition.StimulusId,
                definition.StimulusCategory,
                definition.ExpectedResponse,
                TaskElapsedMs,
                _trial.ElapsedSinceStimulusMs(_nowMs),
                RuntimeState.DifficultyLevel);
            EmitTaskEvent(detail);
            EmitTaskEvent(Telemetry.TrialCompleted(detail));

            CompletedTrialCount++;
            TimeoutCount++;
            CurrentStimulusCategory = null;
            LastFeedback = "no response";
            ScheduleNextTrial();
        }

        private void ScheduleNextTrial()
        {
            _feedbackUntilMs = _nowMs + feedbackDurationMs;
            _nextTrialAtMs = _feedbackUntilMs + RuntimeState.StimulusIntervalMs;
        }

        // -- protocol -------------------------------------------------------

        private void EmitTaskEvent(TaskEventDetail detail)
        {
            Transport.Send(Telemetry.TaskEvent(detail));
        }

        private void EmitTaskState()
        {
            Transport.Send(Telemetry.TaskState(new TaskStatePayload
            {
                State = RuntimeState.State,
                TaskId = taskId,
                BlockId = RuntimeState.BlockId,
                TrialId = RuntimeState.TrialId,
                DifficultyLevel = RuntimeState.DifficultyLevel,
                StimulusIntervalMs = RuntimeState.StimulusIntervalMs
            }));
        }

        private void SendHeartbeatIfDue()
        {
            if (_nowMs - _lastHeartbeatMs < _heartbeatIntervalMs)
            {
                return;
            }

            _lastHeartbeatMs = _nowMs;
            _heartbeatCounter++;
            Transport.Send(
                Telemetry.Heartbeat("hb-" + _heartbeatCounter, _nowMs / 1000.0));
        }

        private void OnMessageReceived(ProtocolEnvelope envelope)
        {
            switch (envelope.MessageTypeValue)
            {
                case MessageType.ServerHello:
                {
                    ServerHelloPayload hello = ServerHelloPayload.FromJson(envelope.Payload);
                    _heartbeatIntervalMs = hello.HeartbeatIntervalSeconds * 1000.0;
                    Debug.Log(
                        "[EngageVR] handshake accepted by " + hello.ServerName
                        + " (protocol " + hello.ProtocolVersionValue + ")");
                    break;
                }

                case MessageType.AdaptationCommand:
                {
                    HandleAdaptationCommand(envelope);
                    break;
                }

                case MessageType.ProtocolError:
                {
                    ProtocolErrorPayload error = ProtocolErrorPayload.FromJson(envelope.Payload);
                    Debug.LogWarning(
                        "[EngageVR] protocol error " + error.ErrorCode + ": " + error.Detail);
                    break;
                }
            }
        }

        private void HandleAdaptationCommand(ProtocolEnvelope envelope)
        {
            AdaptationCommandPayload command;
            try
            {
                command = AdaptationCommandPayload.FromJson(envelope.Payload);
            }
            catch (JsonException exception)
            {
                Debug.LogWarning(
                    "[EngageVR] rejected a malformed adaptation command: "
                    + exception.Message);
                return;
            }

            AdaptationAcknowledgementPayload acknowledgement =
                _adaptation.Apply(command, DateTime.UtcNow);
            Transport.Send(
                Telemetry.AdaptationAcknowledgement(acknowledgement, envelope.MessageId));

            if (!acknowledgement.Accepted)
            {
                return;
            }

            if (string.Equals(
                    command.Command, AdaptationCommandName.PauseTask, StringComparison.Ordinal))
            {
                EmitTaskEvent(
                    Telemetry.Lifecycle(
                        TaskEventType.TaskPaused, TaskElapsedMs, RuntimeState.DifficultyLevel));
            }
            else if (string.Equals(
                         command.Command,
                         AdaptationCommandName.ResumeTask,
                         StringComparison.Ordinal))
            {
                EmitTaskEvent(
                    Telemetry.Lifecycle(
                        TaskEventType.TaskResumed, TaskElapsedMs, RuntimeState.DifficultyLevel));
            }

            EmitTaskState();
        }

        private void OnTransportFaulted(string detail)
        {
            Debug.LogWarning("[EngageVR] transport fault: " + detail);
        }
    }
}
