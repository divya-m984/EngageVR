// EngageVR -- the protocol envelope, mirroring engagevr/protocol/envelope.py.
//
// Field names here are the wire contract. They must match the Python
// models exactly; the shared fixtures under protocol/fixtures/ are
// parsed by both test suites so a rename on either side fails the
// other.

using System;
using System.Collections.Generic;
using System.Globalization;

namespace EngageVR.Protocol
{
    /// <summary>Protocol version constants. Mirrors protocol/version.py.</summary>
    public static class ProtocolVersion
    {
        public const string Current = "1.0";

        public static readonly int[] AcceptedMajorVersions = { 1 };

        /// <summary>Parse "MAJOR.MINOR". Returns false on a malformed value.</summary>
        public static bool TryParse(string value, out int major, out int minor)
        {
            major = 0;
            minor = 0;
            if (string.IsNullOrEmpty(value))
            {
                return false;
            }

            string[] parts = value.Trim().Split('.');
            if (parts.Length != 2)
            {
                return false;
            }

            return int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out major)
                   && int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out minor);
        }

        /// <summary>
        /// Whether this build can parse a message declaring
        /// <paramref name="value"/>. A different major version is
        /// rejected outright rather than parsed on a best-effort basis.
        /// </summary>
        public static bool IsSupported(string value)
        {
            int major;
            int minor;
            if (!TryParse(value, out major, out minor))
            {
                return false;
            }

            foreach (int accepted in AcceptedMajorVersions)
            {
                if (accepted == major)
                {
                    return true;
                }
            }

            return false;
        }
    }

    /// <summary>Message type names. Mirrors MessageType in Python.</summary>
    public static class MessageType
    {
        public const string ClientHello = "client_hello";
        public const string ServerHello = "server_hello";
        public const string SessionStart = "session_start";
        public const string SessionEnd = "session_end";
        public const string TaskEvent = "task_event";
        public const string TaskState = "task_state";
        public const string Telemetry = "telemetry";
        public const string AdaptationCommand = "adaptation_command";
        public const string AdaptationAcknowledgement = "adaptation_acknowledgement";
        public const string Heartbeat = "heartbeat";
        public const string HeartbeatAcknowledgement = "heartbeat_acknowledgement";
        public const string ReplayControl = "replay_control";
        public const string Acknowledgement = "acknowledgement";
        public const string ProtocolError = "protocol_error";

        public static readonly string[] All =
        {
            ClientHello, ServerHello, SessionStart, SessionEnd, TaskEvent,
            TaskState, Telemetry, AdaptationCommand, AdaptationAcknowledgement,
            Heartbeat, HeartbeatAcknowledgement, ReplayControl, Acknowledgement,
            ProtocolError
        };

        public static bool IsKnown(string value)
        {
            foreach (string known in All)
            {
                if (string.Equals(known, value, StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }
    }

    /// <summary>Message source names. Mirrors MessageSource in Python.</summary>
    public static class MessageSource
    {
        public const string PythonSimulator = "python_simulator";
        public const string UnityClient = "unity_client";
        public const string Backend = "backend";
        public const string Replay = "replay";
        public const string TestFixture = "test_fixture";
    }

    /// <summary>Client role names. Mirrors ClientRole in Python.</summary>
    public static class ClientRole
    {
        public const string Simulator = "simulator";
        public const string Unity = "unity";
        public const string Observer = "observer";
        public const string Replay = "replay";
    }

    /// <summary>Data-source and label constants.</summary>
    public static class DataSource
    {
        public const string Synthetic = "synthetic";
        public const string PublicDataset = "public_dataset";
        public const string Live = "live";

        /// <summary>Permanent marker on generated content.</summary>
        public const string SyntheticLabel = "SYNTHETIC";

        /// <summary>Permanent marker added by the replay player.</summary>
        public const string ReplayLabel = "REPLAY";
    }

    /// <summary>
    /// Where a message's content came from.
    /// <para>
    /// A synthetic message must carry
    /// <see cref="DataSource.SyntheticLabel"/>; the Python receiver
    /// rejects a synthetic message without it.
    /// </para>
    /// </summary>
    public sealed class MessageProvenance
    {
        public string DataSource = EngageVR.Protocol.DataSource.Live;
        public string SyntheticLabel;
        public string Producer = string.Empty;

        public static MessageProvenance Live(string producer)
        {
            return new MessageProvenance
            {
                DataSource = EngageVR.Protocol.DataSource.Live,
                SyntheticLabel = null,
                Producer = producer ?? string.Empty
            };
        }

        public static MessageProvenance Synthetic(string producer)
        {
            return new MessageProvenance
            {
                DataSource = EngageVR.Protocol.DataSource.Synthetic,
                SyntheticLabel = EngageVR.Protocol.DataSource.SyntheticLabel,
                Producer = producer ?? string.Empty
            };
        }

        public JsonValue ToJson()
        {
            return JsonValue.Object()
                .Set("data_source", JsonValue.String(DataSource))
                .Set("synthetic_label", Json.OrNull(SyntheticLabel))
                .Set("producer", JsonValue.String(Producer ?? string.Empty));
        }

        public static MessageProvenance FromJson(JsonValue value)
        {
            if (value == null || value.IsNull)
            {
                return Live(string.Empty);
            }

            return new MessageProvenance
            {
                DataSource = value.GetStringOrNull("data_source")
                             ?? EngageVR.Protocol.DataSource.Live,
                SyntheticLabel = value.GetStringOrNull("synthetic_label"),
                Producer = value.GetStringOrNull("producer") ?? string.Empty
            };
        }
    }

    /// <summary>
    /// Metadata added when a recorded message is replayed. Present only
    /// on replayed messages; its presence is what marks a message as a
    /// replay rather than a live message.
    /// </summary>
    public sealed class ReplayMetadata
    {
        public string ReplayLabel = DataSource.ReplayLabel;
        public string SourceSessionId;
        public string ReplaySessionId;
        public int ReplayIndex;
        public double ReplaySpeed;
        public string ReplayedAtUtc;
        public int? OriginalArrivalIndex;

        public JsonValue ToJson()
        {
            return JsonValue.Object()
                .Set("replay_label", JsonValue.String(ReplayLabel))
                .Set("source_session_id", JsonValue.String(SourceSessionId))
                .Set("replay_session_id", JsonValue.String(ReplaySessionId))
                .Set("replay_index", JsonValue.Number(ReplayIndex))
                .Set("replay_speed", JsonValue.Number(ReplaySpeed))
                .Set("replayed_at_utc", JsonValue.String(ReplayedAtUtc))
                .Set("original_arrival_index", Json.OrNull(OriginalArrivalIndex));
        }

        public static ReplayMetadata FromJson(JsonValue value)
        {
            if (value == null || value.IsNull)
            {
                return null;
            }

            return new ReplayMetadata
            {
                ReplayLabel = value.GetStringOrNull("replay_label") ?? DataSource.ReplayLabel,
                SourceSessionId = value.GetStringOrNull("source_session_id"),
                ReplaySessionId = value.GetStringOrNull("replay_session_id"),
                ReplayIndex = value.GetIntOrNull("replay_index") ?? 0,
                ReplaySpeed = value.GetDoubleOrNull("replay_speed") ?? 0.0,
                ReplayedAtUtc = value.GetStringOrNull("replayed_at_utc"),
                OriginalArrivalIndex = value.GetIntOrNull("original_arrival_index")
            };
        }
    }

    /// <summary>
    /// One protocol message: identity, ordering, timing, provenance, and
    /// a payload held as a parsed JSON object.
    /// </summary>
    public sealed class ProtocolEnvelope
    {
        public string ProtocolVersionValue = ProtocolVersion.Current;
        public string MessageId;
        public string MessageTypeValue;
        public string SessionId;
        public string Source = MessageSource.UnityClient;
        public int SequenceNumber;

        /// <summary>Sender wall clock, ISO-8601 with a UTC offset.</summary>
        public string SentAtUtc;

        /// <summary>
        /// Sender's own monotonic clock. Its origin is arbitrary and it
        /// is only comparable with other readings from this same client.
        /// The backend records it without translating it.
        /// </summary>
        public double SentAtMonotonicSeconds;

        public JsonValue Payload = JsonValue.Object();
        public string CorrelationId;
        public MessageProvenance Provenance = MessageProvenance.Live(string.Empty);
        public ReplayMetadata Replay;

        public JsonValue ToJson()
        {
            JsonValue root = JsonValue.Object()
                .Set("protocol_version", JsonValue.String(ProtocolVersionValue))
                .Set("message_id", JsonValue.String(MessageId))
                .Set("message_type", JsonValue.String(MessageTypeValue))
                .Set("session_id", JsonValue.String(SessionId))
                .Set("source", JsonValue.String(Source))
                .Set("sequence_number", JsonValue.Number(SequenceNumber))
                .Set("sent_at_utc", JsonValue.String(SentAtUtc))
                .Set("sent_at_monotonic_seconds", JsonValue.Number(SentAtMonotonicSeconds))
                .Set("payload", Payload ?? JsonValue.Object())
                .Set("correlation_id", Json.OrNull(CorrelationId))
                .Set("provenance", Provenance.ToJson());

            root.Set("replay", Replay == null ? JsonValue.Null : Replay.ToJson());
            return root;
        }

        public string ToJsonString()
        {
            return Json.Serialize(ToJson());
        }

        public static ProtocolEnvelope FromJson(JsonValue value)
        {
            if (value == null || value.Kind != JsonKind.Object)
            {
                throw new JsonException("a protocol message must be a JSON object");
            }

            string version = value.GetRequiredString("protocol_version");
            if (!ProtocolVersion.IsSupported(version))
            {
                throw new JsonException(
                    "unsupported protocol version '" + version + "'; this client speaks "
                    + ProtocolVersion.Current);
            }

            string messageType = value.GetRequiredString("message_type");
            if (!MessageType.IsKnown(messageType))
            {
                throw new JsonException("unknown message_type '" + messageType + "'");
            }

            return new ProtocolEnvelope
            {
                ProtocolVersionValue = version,
                MessageId = value.GetRequiredString("message_id"),
                MessageTypeValue = messageType,
                SessionId = value.GetRequiredString("session_id"),
                Source = value.GetRequiredString("source"),
                SequenceNumber = value.GetRequiredInt("sequence_number"),
                SentAtUtc = value.GetRequiredString("sent_at_utc"),
                SentAtMonotonicSeconds = value.GetRequiredDouble("sent_at_monotonic_seconds"),
                Payload = value.Get("payload"),
                CorrelationId = value.GetStringOrNull("correlation_id"),
                Provenance = MessageProvenance.FromJson(value.Get("provenance")),
                Replay = ReplayMetadata.FromJson(value.Get("replay"))
            };
        }

        public static ProtocolEnvelope Parse(string text)
        {
            return FromJson(Json.Parse(text));
        }

        /// <summary>Format a UTC instant the way the Python side expects.</summary>
        public static string FormatUtc(DateTime instant)
        {
            return instant.ToUniversalTime()
                .ToString("yyyy-MM-ddTHH:mm:ss.ffffffK", CultureInfo.InvariantCulture);
        }
    }

    /// <summary>Builds envelopes with a per-source sequence counter.</summary>
    public sealed class EnvelopeFactory
    {
        private readonly string _sessionId;
        private readonly string _source;
        private readonly MessageProvenance _provenance;
        private readonly Func<double> _monotonic;
        private int _sequence;

        public EnvelopeFactory(
            string sessionId,
            string source,
            MessageProvenance provenance,
            Func<double> monotonic)
        {
            _sessionId = sessionId;
            _source = source;
            _provenance = provenance;
            _monotonic = monotonic ?? (() => 0.0);
        }

        public int NextSequenceNumber
        {
            get { return _sequence; }
        }

        public ProtocolEnvelope Build(string messageType, JsonValue payload, string correlationId)
        {
            ProtocolEnvelope envelope = new ProtocolEnvelope
            {
                ProtocolVersionValue = ProtocolVersion.Current,
                MessageId = Guid.NewGuid().ToString("N"),
                MessageTypeValue = messageType,
                SessionId = _sessionId,
                Source = _source,
                SequenceNumber = _sequence,
                SentAtUtc = ProtocolEnvelope.FormatUtc(DateTime.UtcNow),
                SentAtMonotonicSeconds = _monotonic(),
                Payload = payload ?? JsonValue.Object(),
                CorrelationId = correlationId,
                Provenance = _provenance,
                Replay = null
            };
            _sequence++;
            return envelope;
        }

        public ProtocolEnvelope Build(string messageType, JsonValue payload)
        {
            return Build(messageType, payload, null);
        }
    }

    /// <summary>Helpers shared by the message payload classes.</summary>
    internal static class PayloadUtility
    {
        public static JsonValue StringList(List<string> values)
        {
            List<JsonValue> items = new List<JsonValue>();
            if (values != null)
            {
                foreach (string value in values)
                {
                    items.Add(JsonValue.String(value));
                }
            }

            return JsonValue.Array(items);
        }
    }
}
