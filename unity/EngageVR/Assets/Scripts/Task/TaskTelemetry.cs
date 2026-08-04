// EngageVR -- builds task-event messages.
//
// SCOPE
// -----
// Everything this class produces is SOFTWARE TELEMETRY: what the task
// program observed about its own trials. Accuracy and reaction time
// here are not engagement measurements, not attention measurements, not
// cognitive-load measurements, and not fatigue measurements. This task
// has not been experimentally designed, piloted, or approved.
//
// The class is deliberately free of UnityEngine types so the EditMode
// tests can construct it directly and compare its output against the
// shared Python fixtures.

using System;
using System.Collections.Generic;

namespace EngageVR.Task
{
    using EngageVR.Protocol;

    /// <summary>Turns task observations into protocol envelopes.</summary>
    public sealed class TaskTelemetry
    {
        private readonly EnvelopeFactory _factory;
        private readonly string _taskId;

        public TaskTelemetry(EnvelopeFactory factory, string taskId)
        {
            if (factory == null)
            {
                throw new ArgumentNullException("factory");
            }

            _factory = factory;
            _taskId = taskId;
        }

        public int EmittedCount { get; private set; }

        public ProtocolEnvelope ClientHello(string clientName, string clientVersion)
        {
            ClientHelloPayload payload = new ClientHelloPayload
            {
                Role = ClientRole.Unity,
                ClientName = clientName,
                ClientVersion = clientVersion,
                ProtocolVersionValue = ProtocolVersion.Current,
                Capabilities = new List<string>
                {
                    "task_event", "adaptation_acknowledgement", "heartbeat"
                }
            };
            return Track(_factory.Build(MessageType.ClientHello, payload.ToJson()));
        }

        public ProtocolEnvelope SessionStart(
            string participantId,
            int blocks,
            int trialsPerBlock,
            int difficultyLevel,
            Dictionary<string, JsonValue> configuration)
        {
            SessionStartPayload payload = new SessionStartPayload
            {
                ParticipantId = participantId,
                TaskId = _taskId,
                StartedAtUtc = ProtocolEnvelope.FormatUtc(DateTime.UtcNow),
                Blocks = blocks,
                TrialsPerBlock = trialsPerBlock,
                DifficultyLevel = difficultyLevel,
                Configuration = configuration ?? new Dictionary<string, JsonValue>()
            };
            return Track(_factory.Build(MessageType.SessionStart, payload.ToJson()));
        }

        public ProtocolEnvelope SessionEnd(bool completed, string reason)
        {
            SessionEndPayload payload = new SessionEndPayload
            {
                EndedAtUtc = ProtocolEnvelope.FormatUtc(DateTime.UtcNow),
                Completed = completed,
                Reason = reason
            };
            return Track(_factory.Build(MessageType.SessionEnd, payload.ToJson()));
        }

        public ProtocolEnvelope TaskEvent(TaskEventDetail detail)
        {
            if (detail == null)
            {
                throw new ArgumentNullException("detail");
            }

            return Track(_factory.Build(MessageType.TaskEvent, detail.ToPayload()));
        }

        public ProtocolEnvelope TaskState(TaskStatePayload payload)
        {
            return Track(_factory.Build(MessageType.TaskState, payload.ToJson()));
        }

        public ProtocolEnvelope Heartbeat(string heartbeatId, double clientMonotonicSeconds)
        {
            HeartbeatPayload payload = new HeartbeatPayload
            {
                HeartbeatId = heartbeatId,
                ClientMonotonicSeconds = clientMonotonicSeconds
            };
            return Track(_factory.Build(MessageType.Heartbeat, payload.ToJson()));
        }

        public ProtocolEnvelope AdaptationAcknowledgement(
            AdaptationAcknowledgementPayload payload, string correlationId)
        {
            return Track(
                _factory.Build(
                    MessageType.AdaptationAcknowledgement, payload.ToJson(), correlationId));
        }

        public ProtocolEnvelope Telemetry(string component, Dictionary<string, double> metrics)
        {
            TelemetryPayload payload = new TelemetryPayload
            {
                Component = component,
                Metrics = metrics ?? new Dictionary<string, double>()
            };
            return Track(_factory.Build(MessageType.Telemetry, payload.ToJson()));
        }

        // -- task event helpers -------------------------------------------

        public TaskEventDetail Lifecycle(string eventType, double taskElapsedMs, int difficulty)
        {
            return new TaskEventDetail
            {
                EventType = eventType,
                TaskId = _taskId,
                DifficultyLevel = difficulty,
                TaskElapsedMs = taskElapsedMs
            };
        }

        public TaskEventDetail BlockEvent(
            string eventType, int blockId, double taskElapsedMs, int difficulty)
        {
            return new TaskEventDetail
            {
                EventType = eventType,
                TaskId = _taskId,
                BlockId = blockId,
                DifficultyLevel = difficulty,
                TaskElapsedMs = taskElapsedMs
            };
        }

        public TaskEventDetail TrialStarted(
            int blockId, int trialId, double taskElapsedMs, int difficulty)
        {
            return new TaskEventDetail
            {
                EventType = TaskEventType.TrialStarted,
                TaskId = _taskId,
                BlockId = blockId,
                TrialId = trialId,
                DifficultyLevel = difficulty,
                TaskElapsedMs = taskElapsedMs,
                TrialElapsedMs = 0.0
            };
        }

        public TaskEventDetail StimulusPresented(
            int blockId,
            int trialId,
            string stimulusId,
            string stimulusCategory,
            string expectedResponse,
            double taskElapsedMs,
            double trialElapsedMs,
            int difficulty)
        {
            return new TaskEventDetail
            {
                EventType = TaskEventType.StimulusPresented,
                TaskId = _taskId,
                BlockId = blockId,
                TrialId = trialId,
                StimulusId = stimulusId,
                StimulusCategory = stimulusCategory,
                ExpectedResponse = expectedResponse,
                DifficultyLevel = difficulty,
                TaskElapsedMs = taskElapsedMs,
                TrialElapsedMs = trialElapsedMs
            };
        }

        /// <summary>
        /// A registered response. <paramref name="reactionTimeMs"/> must
        /// be non-negative; a caller that has no reaction time must emit
        /// <see cref="ResponseTimeout"/> instead of passing 0.
        /// </summary>
        public TaskEventDetail ResponseRegistered(
            int blockId,
            int trialId,
            string stimulusId,
            string stimulusCategory,
            string expectedResponse,
            string observedResponse,
            bool correct,
            double reactionTimeMs,
            double taskElapsedMs,
            double trialElapsedMs,
            int difficulty)
        {
            if (reactionTimeMs < 0.0)
            {
                throw new ArgumentOutOfRangeException(
                    "reactionTimeMs", "a reaction time may not be negative");
            }

            return new TaskEventDetail
            {
                EventType = TaskEventType.ResponseRegistered,
                TaskId = _taskId,
                BlockId = blockId,
                TrialId = trialId,
                StimulusId = stimulusId,
                StimulusCategory = stimulusCategory,
                ExpectedResponse = expectedResponse,
                ObservedResponse = observedResponse,
                ResponseCorrect = correct,
                ResponseOutcomeValue =
                    correct ? ResponseOutcome.Correct : ResponseOutcome.Incorrect,
                ReactionTimeMs = reactionTimeMs,
                DifficultyLevel = difficulty,
                TaskElapsedMs = taskElapsedMs,
                TrialElapsedMs = trialElapsedMs
            };
        }

        /// <summary>
        /// A trial where no response arrived. Observed response,
        /// correctness, and reaction time are all left null: a timeout
        /// is a distinct observation from a wrong response, and it is
        /// never recorded as a zero-millisecond response.
        /// </summary>
        public TaskEventDetail ResponseTimeout(
            int blockId,
            int trialId,
            string stimulusId,
            string stimulusCategory,
            string expectedResponse,
            double taskElapsedMs,
            double trialElapsedMs,
            int difficulty)
        {
            return new TaskEventDetail
            {
                EventType = TaskEventType.ResponseTimeout,
                TaskId = _taskId,
                BlockId = blockId,
                TrialId = trialId,
                StimulusId = stimulusId,
                StimulusCategory = stimulusCategory,
                ExpectedResponse = expectedResponse,
                ObservedResponse = null,
                ResponseCorrect = null,
                ResponseOutcomeValue = ResponseOutcome.Timeout,
                ReactionTimeMs = null,
                DifficultyLevel = difficulty,
                TaskElapsedMs = taskElapsedMs,
                TrialElapsedMs = trialElapsedMs
            };
        }

        public TaskEventDetail TrialCompleted(TaskEventDetail resolution)
        {
            TaskEventDetail completed = new TaskEventDetail
            {
                EventType = TaskEventType.TrialCompleted,
                TaskId = _taskId,
                BlockId = resolution.BlockId,
                TrialId = resolution.TrialId,
                StimulusId = resolution.StimulusId,
                StimulusCategory = resolution.StimulusCategory,
                ExpectedResponse = resolution.ExpectedResponse,
                ObservedResponse = resolution.ObservedResponse,
                ResponseCorrect = resolution.ResponseCorrect,
                ResponseOutcomeValue = resolution.ResponseOutcomeValue,
                ReactionTimeMs = resolution.ReactionTimeMs,
                DifficultyLevel = resolution.DifficultyLevel,
                TaskElapsedMs = resolution.TaskElapsedMs,
                TrialElapsedMs = resolution.TrialElapsedMs
            };
            return completed;
        }

        private ProtocolEnvelope Track(ProtocolEnvelope envelope)
        {
            EmittedCount++;
            return envelope;
        }
    }
}
