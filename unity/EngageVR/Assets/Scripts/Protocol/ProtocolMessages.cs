// EngageVR -- typed payloads, mirroring engagevr/protocol/messages.py.
//
// Every nullable field here is genuinely nullable. A missing response
// serializes as JSON null, never as "" or 0: the Python schema treats
// 0 ms as a real reaction time and would accept a fabricated one
// silently, so representing "no response" correctly is a correctness
// requirement, not a style preference.

using System;
using System.Collections.Generic;

namespace EngageVR.Protocol
{
    /// <summary>Task-event names. Mirrors TASK_EVENT_TYPES in Python.</summary>
    public static class TaskEventType
    {
        public const string TaskLoaded = "task_loaded";
        public const string TaskStarted = "task_started";
        public const string BlockStarted = "block_started";
        public const string TrialStarted = "trial_started";
        public const string StimulusPresented = "stimulus_presented";
        public const string ResponseRegistered = "response_registered";
        public const string ResponseTimeout = "response_timeout";
        public const string TrialCompleted = "trial_completed";
        public const string BlockCompleted = "block_completed";
        public const string TaskPaused = "task_paused";
        public const string TaskResumed = "task_resumed";
        public const string TaskCompleted = "task_completed";
        public const string TaskAborted = "task_aborted";
    }

    /// <summary>
    /// How a response slot resolved. Timeout is deliberately distinct
    /// from Incorrect: no response is a different observation from a
    /// wrong response.
    /// </summary>
    public static class ResponseOutcome
    {
        public const string Correct = "correct";
        public const string Incorrect = "incorrect";
        public const string Timeout = "timeout";
    }

    /// <summary>Coarse task lifecycle state.</summary>
    public static class TaskStateName
    {
        public const string Idle = "idle";
        public const string Loaded = "loaded";
        public const string Running = "running";
        public const string Paused = "paused";
        public const string Completed = "completed";
        public const string Aborted = "aborted";
    }

    /// <summary>The visually harmless commands this client implements.</summary>
    public static class AdaptationCommandName
    {
        public const string SetDifficulty = "set_difficulty";
        public const string SetStimulusInterval = "set_stimulus_interval";
        public const string PauseTask = "pause_task";
        public const string ResumeTask = "resume_task";
    }

    /// <summary>The client's opening handshake message.</summary>
    public sealed class ClientHelloPayload
    {
        public string Role = ClientRole.Unity;
        public string ClientName = "engagevr-unity-client";
        public string ClientVersion = "0.1.0";
        public string ProtocolVersionValue = ProtocolVersion.Current;
        public List<string> Capabilities = new List<string>();

        public JsonValue ToJson()
        {
            return JsonValue.Object()
                .Set("role", JsonValue.String(Role))
                .Set("client_name", JsonValue.String(ClientName))
                .Set("client_version", JsonValue.String(ClientVersion))
                .Set("protocol_version", JsonValue.String(ProtocolVersionValue))
                .Set("capabilities", PayloadUtility.StringList(Capabilities));
        }
    }

    /// <summary>The backend's handshake response.</summary>
    public sealed class ServerHelloPayload
    {
        public bool Accepted;
        public string ServerName;
        public string ServerVersion;
        public string ProtocolVersionValue;
        public string SessionId;
        public string AssignedClientId;
        public string ServerTimeUtc;
        public double HeartbeatIntervalSeconds;
        public double ConnectionTimeoutSeconds;
        public int MaximumMessageBytes;
        public string RejectionReason;

        public static ServerHelloPayload FromJson(JsonValue value)
        {
            return new ServerHelloPayload
            {
                Accepted = value.GetBoolOrNull("accepted") ?? false,
                ServerName = value.GetStringOrNull("server_name"),
                ServerVersion = value.GetStringOrNull("server_version"),
                ProtocolVersionValue = value.GetStringOrNull("protocol_version"),
                SessionId = value.GetStringOrNull("session_id"),
                AssignedClientId = value.GetStringOrNull("assigned_client_id"),
                ServerTimeUtc = value.GetStringOrNull("server_time_utc"),
                HeartbeatIntervalSeconds = value.GetDoubleOrNull("heartbeat_interval_seconds") ?? 10.0,
                ConnectionTimeoutSeconds = value.GetDoubleOrNull("connection_timeout_seconds") ?? 30.0,
                MaximumMessageBytes = value.GetIntOrNull("maximum_message_bytes") ?? 262144,
                RejectionReason = value.GetStringOrNull("rejection_reason")
            };
        }
    }

    /// <summary>Declares the start of a task session.</summary>
    public sealed class SessionStartPayload
    {
        /// <summary>Pseudonymous label only. Never a real-world identity.</summary>
        public string ParticipantId;

        public string TaskId;
        public string StartedAtUtc;
        public int Blocks;
        public int TrialsPerBlock;
        public int DifficultyLevel;
        public Dictionary<string, JsonValue> Configuration = new Dictionary<string, JsonValue>();

        public JsonValue ToJson()
        {
            JsonValue configuration = JsonValue.Object();
            foreach (KeyValuePair<string, JsonValue> pair in Configuration)
            {
                configuration.Set(pair.Key, pair.Value);
            }

            return JsonValue.Object()
                .Set("participant_id", JsonValue.String(ParticipantId))
                .Set("task_id", JsonValue.String(TaskId))
                .Set("started_at_utc", JsonValue.String(StartedAtUtc))
                .Set("blocks", JsonValue.Number(Blocks))
                .Set("trials_per_block", JsonValue.Number(TrialsPerBlock))
                .Set("difficulty_level", JsonValue.Number(DifficultyLevel))
                .Set("configuration", configuration);
        }
    }

    /// <summary>Declares the end of a task session.</summary>
    public sealed class SessionEndPayload
    {
        public string EndedAtUtc;
        public bool Completed;
        public string Reason;

        public JsonValue ToJson()
        {
            return JsonValue.Object()
                .Set("ended_at_utc", JsonValue.String(EndedAtUtc))
                .Set("completed", JsonValue.Bool(Completed))
                .Set("reason", JsonValue.String(Reason));
        }
    }

    /// <summary>
    /// One task event.
    /// <para>
    /// Every nullable field defaults to null and serializes as JSON
    /// null. In particular <see cref="ObservedResponse"/>,
    /// <see cref="ResponseCorrect"/>, and
    /// <see cref="ReactionTimeMs"/> stay null when no response was
    /// registered.
    /// </para>
    /// <para>
    /// This is a software telemetry record. It is not a measurement of
    /// engagement, attention, cognitive load, or fatigue.
    /// </para>
    /// </summary>
    public sealed class TaskEventDetail
    {
        public string EventType;
        public string TaskId;
        public int? BlockId;
        public int? TrialId;
        public string StimulusId;
        public string StimulusCategory;
        public string ExpectedResponse;
        public string ObservedResponse;
        public bool? ResponseCorrect;
        public string ResponseOutcomeValue;
        public double? ReactionTimeMs;
        public int? DifficultyLevel;
        public double? TaskElapsedMs;
        public double? TrialElapsedMs;

        public JsonValue ToJson()
        {
            if (ReactionTimeMs.HasValue && ReactionTimeMs.Value < 0.0)
            {
                throw new JsonException(
                    "reaction_time_ms must not be negative; the Python receiver "
                    + "rejects the message");
            }

            return JsonValue.Object()
                .Set("event_type", JsonValue.String(EventType))
                .Set("task_id", Json.OrNull(TaskId))
                .Set("block_id", Json.OrNull(BlockId))
                .Set("trial_id", Json.OrNull(TrialId))
                .Set("stimulus_id", Json.OrNull(StimulusId))
                .Set("stimulus_category", Json.OrNull(StimulusCategory))
                .Set("expected_response", Json.OrNull(ExpectedResponse))
                .Set("observed_response", Json.OrNull(ObservedResponse))
                .Set("response_correct", Json.OrNull(ResponseCorrect))
                .Set("response_outcome", Json.OrNull(ResponseOutcomeValue))
                .Set("reaction_time_ms", Json.OrNull(ReactionTimeMs))
                .Set("difficulty_level", Json.OrNull(DifficultyLevel))
                .Set("task_elapsed_ms", Json.OrNull(TaskElapsedMs))
                .Set("trial_elapsed_ms", Json.OrNull(TrialElapsedMs));
        }

        public static TaskEventDetail FromJson(JsonValue value)
        {
            return new TaskEventDetail
            {
                EventType = value.GetRequiredString("event_type"),
                TaskId = value.GetStringOrNull("task_id"),
                BlockId = value.GetIntOrNull("block_id"),
                TrialId = value.GetIntOrNull("trial_id"),
                StimulusId = value.GetStringOrNull("stimulus_id"),
                StimulusCategory = value.GetStringOrNull("stimulus_category"),
                ExpectedResponse = value.GetStringOrNull("expected_response"),
                ObservedResponse = value.GetStringOrNull("observed_response"),
                ResponseCorrect = value.GetBoolOrNull("response_correct"),
                ResponseOutcomeValue = value.GetStringOrNull("response_outcome"),
                ReactionTimeMs = value.GetDoubleOrNull("reaction_time_ms"),
                DifficultyLevel = value.GetIntOrNull("difficulty_level"),
                TaskElapsedMs = value.GetDoubleOrNull("task_elapsed_ms"),
                TrialElapsedMs = value.GetDoubleOrNull("trial_elapsed_ms")
            };
        }

        /// <summary>Wrap this detail in a task_event payload object.</summary>
        public JsonValue ToPayload()
        {
            return JsonValue.Object().Set("event", ToJson());
        }

        public static TaskEventDetail FromPayload(JsonValue payload)
        {
            return FromJson(payload.Get("event"));
        }
    }

    /// <summary>Current coarse state of the task client.</summary>
    public sealed class TaskStatePayload
    {
        public string State = TaskStateName.Idle;
        public string TaskId;
        public int? BlockId;
        public int? TrialId;
        public int? DifficultyLevel;
        public double? StimulusIntervalMs;

        public JsonValue ToJson()
        {
            return JsonValue.Object()
                .Set("state", JsonValue.String(State))
                .Set("task_id", Json.OrNull(TaskId))
                .Set("block_id", Json.OrNull(BlockId))
                .Set("trial_id", Json.OrNull(TrialId))
                .Set("difficulty_level", Json.OrNull(DifficultyLevel))
                .Set("stimulus_interval_ms", Json.OrNull(StimulusIntervalMs));
        }
    }

    /// <summary>
    /// Software/runtime telemetry: frame rate, dropped frames, and
    /// similar program measurements. It cannot carry behavioural,
    /// physiological, engagement, or cognitive-load values.
    /// </summary>
    public sealed class TelemetryPayload
    {
        public string Component;
        public Dictionary<string, double> Metrics = new Dictionary<string, double>();

        public JsonValue ToJson()
        {
            JsonValue metrics = JsonValue.Object();
            foreach (KeyValuePair<string, double> pair in Metrics)
            {
                metrics.Set(pair.Key, JsonValue.Number(pair.Value));
            }

            return JsonValue.Object()
                .Set("component", JsonValue.String(Component))
                .Set("metrics", metrics);
        }
    }

    /// <summary>An adaptation command received from the backend.</summary>
    public sealed class AdaptationCommandPayload
    {
        public string CommandId;
        public string Command;
        public JsonValue Value = JsonValue.Null;
        public string Reason;
        public string IssuedAtUtc;
        public string ExpiresAtUtc;
        public string TargetRole;
        public string TargetClientId;
        public bool IsManual = true;

        public static AdaptationCommandPayload FromJson(JsonValue value)
        {
            return new AdaptationCommandPayload
            {
                CommandId = value.GetRequiredString("command_id"),
                Command = value.GetRequiredString("command"),
                Value = value.Get("value"),
                Reason = value.GetStringOrNull("reason"),
                IssuedAtUtc = value.GetStringOrNull("issued_at_utc"),
                ExpiresAtUtc = value.GetStringOrNull("expires_at_utc"),
                TargetRole = value.GetStringOrNull("target_role"),
                TargetClientId = value.GetStringOrNull("target_client_id"),
                IsManual = value.GetBoolOrNull("is_manual") ?? true
            };
        }

        /// <summary>
        /// Whether this command has already expired at
        /// <paramref name="nowUtc"/>. Returns false when no expiry was
        /// set; an unparseable expiry is treated as expired, because a
        /// command whose deadline cannot be read must not be applied.
        /// </summary>
        public bool IsExpired(DateTime nowUtc)
        {
            if (string.IsNullOrEmpty(ExpiresAtUtc))
            {
                return false;
            }

            DateTime parsed;
            if (!DateTime.TryParse(
                    ExpiresAtUtc,
                    System.Globalization.CultureInfo.InvariantCulture,
                    System.Globalization.DateTimeStyles.AdjustToUniversal
                    | System.Globalization.DateTimeStyles.AssumeUniversal,
                    out parsed))
            {
                return true;
            }

            return nowUtc > parsed;
        }
    }

    /// <summary>This client's response to an adaptation command.</summary>
    public sealed class AdaptationAcknowledgementPayload
    {
        public string CommandId;
        public bool Accepted;
        public string AppliedAtUtc;
        public string RejectionReason;
        public bool Duplicate;

        public JsonValue ToJson()
        {
            return JsonValue.Object()
                .Set("command_id", JsonValue.String(CommandId))
                .Set("accepted", JsonValue.Bool(Accepted))
                .Set("applied_at_utc", Json.OrNull(AppliedAtUtc))
                .Set("rejection_reason", Json.OrNull(RejectionReason))
                .Set("duplicate", JsonValue.Bool(Duplicate));
        }
    }

    /// <summary>Liveness probe.</summary>
    public sealed class HeartbeatPayload
    {
        public string HeartbeatId;
        public double ClientMonotonicSeconds;

        public JsonValue ToJson()
        {
            return JsonValue.Object()
                .Set("heartbeat_id", JsonValue.String(HeartbeatId))
                .Set("client_monotonic_seconds", JsonValue.Number(ClientMonotonicSeconds));
        }
    }

    /// <summary>A typed rejection notice from the backend.</summary>
    public sealed class ProtocolErrorPayload
    {
        public string ErrorCode;
        public string Detail;
        public string OffendingMessageId;
        public string OffendingMessageType;
        public int? OffendingSequenceNumber;
        public bool Fatal;

        public static ProtocolErrorPayload FromJson(JsonValue value)
        {
            return new ProtocolErrorPayload
            {
                ErrorCode = value.GetStringOrNull("error_code"),
                Detail = value.GetStringOrNull("detail"),
                OffendingMessageId = value.GetStringOrNull("offending_message_id"),
                OffendingMessageType = value.GetStringOrNull("offending_message_type"),
                OffendingSequenceNumber = value.GetIntOrNull("offending_sequence_number"),
                Fatal = value.GetBoolOrNull("fatal") ?? false
            };
        }
    }

    /// <summary>The backend's per-message acknowledgement.</summary>
    public sealed class AcknowledgementPayload
    {
        public string AcknowledgedMessageId;
        public string AcknowledgedMessageType;
        public int AcknowledgedSequenceNumber;
        public string ServerReceivedAtUtc;
        public bool Stored;
        public bool Dropped;

        public static AcknowledgementPayload FromJson(JsonValue value)
        {
            return new AcknowledgementPayload
            {
                AcknowledgedMessageId = value.GetStringOrNull("acknowledged_message_id"),
                AcknowledgedMessageType = value.GetStringOrNull("acknowledged_message_type"),
                AcknowledgedSequenceNumber = value.GetIntOrNull("acknowledged_sequence_number") ?? 0,
                ServerReceivedAtUtc = value.GetStringOrNull("server_received_at_utc"),
                Stored = value.GetBoolOrNull("stored") ?? false,
                Dropped = value.GetBoolOrNull("dropped") ?? false
            };
        }
    }
}
