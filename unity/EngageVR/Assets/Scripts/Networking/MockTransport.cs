// EngageVR -- offline transport.
//
// Lets the Unity task run with no Python process, no server, and no
// network: messages are validated and recorded in memory instead of
// being sent. This is what makes the Unity scene demonstrable and
// testable on its own, and it is also the transport the EditMode and
// PlayMode tests use.
//
// The mock still performs a real handshake: it answers a client_hello
// with a synthesised server_hello, so the task controller's connected
// path is exercised rather than bypassed.

using System;
using System.Collections.Generic;

namespace EngageVR.Networking
{
    using EngageVR.Protocol;

    /// <summary>An in-memory transport for offline runs and tests.</summary>
    public sealed class MockTransport : IMessageTransport
    {
        private readonly Queue<ProtocolEnvelope> _inbound = new Queue<ProtocolEnvelope>();
        private readonly List<ProtocolEnvelope> _sent = new List<ProtocolEnvelope>();
        private readonly bool _answerHandshake;
        private int _serverSequence;

        public MockTransport() : this(true)
        {
        }

        public MockTransport(bool answerHandshake)
        {
            _answerHandshake = answerHandshake;
        }

        /// <summary>Every envelope the client has sent, in order.</summary>
        public IReadOnlyList<ProtocolEnvelope> Sent
        {
            get { return _sent; }
        }

        public TransportState State { get; private set; }

        public string StatusDetail
        {
            get { return "offline mock transport (no server, no network)"; }
        }

        public event Action<ProtocolEnvelope> MessageReceived;

        public event Action<string> Faulted;

        public void Connect()
        {
            State = TransportState.Connected;
        }

        public void Send(ProtocolEnvelope envelope)
        {
            if (envelope == null)
            {
                return;
            }

            // Round-trip through JSON so the mock exercises the same
            // serializer the real transport uses. A payload this client
            // cannot re-parse is a bug worth failing on here rather than
            // discovering against a live backend.
            string text = envelope.ToJsonString();
            ProtocolEnvelope reparsed = ProtocolEnvelope.Parse(text);
            _sent.Add(reparsed);

            if (_answerHandshake
                && string.Equals(
                    reparsed.MessageTypeValue, MessageType.ClientHello, StringComparison.Ordinal))
            {
                EnqueueServerHello(reparsed);
            }
        }

        /// <summary>Queue an inbound message, as though the server sent it.</summary>
        public void Inject(ProtocolEnvelope envelope)
        {
            _inbound.Enqueue(envelope);
        }

        /// <summary>Queue an adaptation command, for offline command testing.</summary>
        public void InjectAdaptationCommand(JsonValue payload, string sessionId)
        {
            Inject(new ProtocolEnvelope
            {
                MessageId = Guid.NewGuid().ToString("N"),
                MessageTypeValue = MessageType.AdaptationCommand,
                SessionId = sessionId,
                Source = MessageSource.Backend,
                SequenceNumber = _serverSequence++,
                SentAtUtc = ProtocolEnvelope.FormatUtc(DateTime.UtcNow),
                SentAtMonotonicSeconds = 0.0,
                Payload = payload,
                Provenance = MessageProvenance.Live("mock_transport")
            });
        }

        public void Poll()
        {
            while (_inbound.Count > 0)
            {
                ProtocolEnvelope envelope = _inbound.Dequeue();
                Action<ProtocolEnvelope> handler = MessageReceived;
                if (handler != null)
                {
                    handler(envelope);
                }
            }
        }

        public void Close()
        {
            State = TransportState.Disconnected;
        }

        public void Dispose()
        {
            Close();
        }

        /// <summary>Report a synthetic fault, for testing the error path.</summary>
        public void SimulateFault(string detail)
        {
            State = TransportState.Faulted;
            Action<string> handler = Faulted;
            if (handler != null)
            {
                handler(detail);
            }
        }

        private void EnqueueServerHello(ProtocolEnvelope hello)
        {
            JsonValue payload = JsonValue.Object()
                .Set("accepted", JsonValue.Bool(true))
                .Set("server_name", JsonValue.String("mock-transport"))
                .Set("server_version", JsonValue.String("0.1.0"))
                .Set("protocol_version", JsonValue.String(ProtocolVersion.Current))
                .Set("session_id", JsonValue.String(hello.SessionId))
                .Set("assigned_client_id", JsonValue.String("offline-client"))
                .Set("server_time_utc", JsonValue.String(ProtocolEnvelope.FormatUtc(DateTime.UtcNow)))
                .Set("heartbeat_interval_seconds", JsonValue.Number(10.0))
                .Set("connection_timeout_seconds", JsonValue.Number(30.0))
                .Set("maximum_message_bytes", JsonValue.Number(262144))
                .Set("rejection_reason", JsonValue.Null);

            Inject(new ProtocolEnvelope
            {
                MessageId = Guid.NewGuid().ToString("N"),
                MessageTypeValue = MessageType.ServerHello,
                SessionId = hello.SessionId,
                Source = MessageSource.Backend,
                SequenceNumber = _serverSequence++,
                SentAtUtc = ProtocolEnvelope.FormatUtc(DateTime.UtcNow),
                SentAtMonotonicSeconds = 0.0,
                Payload = payload,
                CorrelationId = hello.MessageId,
                Provenance = MessageProvenance.Live("mock_transport")
            });
        }
    }
}
