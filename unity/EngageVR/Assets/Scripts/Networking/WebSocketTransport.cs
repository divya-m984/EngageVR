// EngageVR -- WebSocket transport built on System.Net.WebSockets.
//
// WHY ClientWebSocket AND NOT A THIRD-PARTY PACKAGE
// -------------------------------------------------
// System.Net.WebSockets.ClientWebSocket ships with the .NET Standard
// 2.1 base class library that Unity's Mono and IL2CPP backends expose,
// so it needs no package, no manifest entry, and no licence review. It
// works in the Editor and in Windows/macOS/Linux standalone players.
//
// Documented limitation: ClientWebSocket is NOT supported on the WebGL
// player, where the browser owns the socket. EngageVR targets a desktop
// player, so this is not a constraint here; a WebGL build would need a
// JavaScript interop bridge and is out of scope.
//
// Threading: send and receive run on background tasks. Nothing here
// touches the Unity API. Inbound messages are queued and dispatched on
// the main thread by Poll(), because Unity API calls from a background
// thread are undefined behaviour.

using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace EngageVR.Networking
{
    using EngageVR.Protocol;

    /// <summary>A real WebSocket connection to the EngageVR backend.</summary>
    public sealed class WebSocketTransport : IMessageTransport
    {
        private readonly Uri _uri;
        private readonly int _maximumMessageBytes;
        private readonly ConcurrentQueue<ProtocolEnvelope> _inbound =
            new ConcurrentQueue<ProtocolEnvelope>();
        private readonly ConcurrentQueue<string> _faults = new ConcurrentQueue<string>();

        private ClientWebSocket _socket;
        private CancellationTokenSource _cancellation;
        private volatile TransportState _state = TransportState.Disconnected;
        private volatile string _statusDetail = "not connected";

        public WebSocketTransport(string url, int maximumMessageBytes)
        {
            if (string.IsNullOrEmpty(url))
            {
                throw new ArgumentException("a WebSocket url is required", "url");
            }

            _uri = new Uri(url);
            _maximumMessageBytes = maximumMessageBytes > 0 ? maximumMessageBytes : 262144;
        }

        public TransportState State
        {
            get { return _state; }
        }

        public string StatusDetail
        {
            get { return _statusDetail; }
        }

        public event Action<ProtocolEnvelope> MessageReceived;

        public event Action<string> Faulted;

        public void Connect()
        {
            if (_state == TransportState.Connecting || _state == TransportState.Connected)
            {
                return;
            }

            _state = TransportState.Connecting;
            _statusDetail = "connecting to " + _uri;
            _cancellation = new CancellationTokenSource();
            _socket = new ClientWebSocket();
            Task.Run(() => RunAsync(_cancellation.Token));
        }

        public void Send(ProtocolEnvelope envelope)
        {
            if (envelope == null)
            {
                return;
            }

            ClientWebSocket socket = _socket;
            if (socket == null || socket.State != WebSocketState.Open)
            {
                Fault("cannot send " + envelope.MessageTypeValue + ": socket is not open");
                return;
            }

            string text = envelope.ToJsonString();
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            if (bytes.Length > _maximumMessageBytes)
            {
                Fault(
                    "refusing to send a " + bytes.Length + " byte message; the server "
                    + "limit is " + _maximumMessageBytes + " bytes");
                return;
            }

            CancellationTokenSource cancellation = _cancellation;
            CancellationToken token = cancellation == null
                ? CancellationToken.None
                : cancellation.Token;

            Task.Run(async () =>
            {
                try
                {
                    await socket.SendAsync(
                        new ArraySegment<byte>(bytes),
                        WebSocketMessageType.Text,
                        true,
                        token).ConfigureAwait(false);
                }
                catch (Exception exception)
                {
                    Fault("send failed: " + exception.Message);
                }
            });
        }

        public void Poll()
        {
            string fault;
            while (_faults.TryDequeue(out fault))
            {
                Action<string> handler = Faulted;
                if (handler != null)
                {
                    handler(fault);
                }
            }

            ProtocolEnvelope envelope;
            while (_inbound.TryDequeue(out envelope))
            {
                Action<ProtocolEnvelope> handler = MessageReceived;
                if (handler != null)
                {
                    handler(envelope);
                }
            }
        }

        public void Close()
        {
            if (_state == TransportState.Disconnected)
            {
                return;
            }

            _state = TransportState.Closing;
            CancellationTokenSource cancellation = _cancellation;
            if (cancellation != null)
            {
                cancellation.Cancel();
            }

            ClientWebSocket socket = _socket;
            if (socket != null && socket.State == WebSocketState.Open)
            {
                try
                {
                    socket.CloseAsync(
                        WebSocketCloseStatus.NormalClosure,
                        "client closing",
                        CancellationToken.None).Wait(1000);
                }
                catch (Exception)
                {
                    // Closing an already-dead socket is not an error worth
                    // surfacing; the state below records the outcome.
                }
            }

            _state = TransportState.Disconnected;
            _statusDetail = "closed";
        }

        public void Dispose()
        {
            Close();
            ClientWebSocket socket = _socket;
            if (socket != null)
            {
                socket.Dispose();
                _socket = null;
            }

            CancellationTokenSource cancellation = _cancellation;
            if (cancellation != null)
            {
                cancellation.Dispose();
                _cancellation = null;
            }
        }

        private async Task RunAsync(CancellationToken token)
        {
            try
            {
                await _socket.ConnectAsync(_uri, token).ConfigureAwait(false);
                _state = TransportState.Connected;
                _statusDetail = "connected to " + _uri;
                await ReceiveLoopAsync(token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                _state = TransportState.Disconnected;
                _statusDetail = "closed";
            }
            catch (Exception exception)
            {
                Fault("connection failed: " + exception.Message);
            }
        }

        private async Task ReceiveLoopAsync(CancellationToken token)
        {
            byte[] buffer = new byte[8192];
            StringBuilder pending = new StringBuilder();

            while (!token.IsCancellationRequested && _socket.State == WebSocketState.Open)
            {
                WebSocketReceiveResult result;
                try
                {
                    result = await _socket.ReceiveAsync(
                        new ArraySegment<byte>(buffer), token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    return;
                }
                catch (Exception exception)
                {
                    Fault("receive failed: " + exception.Message);
                    return;
                }

                if (result.MessageType == WebSocketMessageType.Close)
                {
                    _state = TransportState.Disconnected;
                    _statusDetail = "server closed the connection";
                    return;
                }

                if (result.MessageType != WebSocketMessageType.Text)
                {
                    // The protocol has no binary representation. Accepting
                    // a binary frame would mean guessing at its meaning.
                    Fault("received a binary frame; the protocol is JSON text only");
                    return;
                }

                pending.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                if (pending.Length > _maximumMessageBytes)
                {
                    Fault(
                        "inbound message exceeded " + _maximumMessageBytes + " bytes");
                    return;
                }

                if (!result.EndOfMessage)
                {
                    continue;
                }

                string text = pending.ToString();
                pending.Length = 0;

                try
                {
                    _inbound.Enqueue(ProtocolEnvelope.Parse(text));
                }
                catch (JsonException exception)
                {
                    // A message this client cannot validate is rejected,
                    // not partially acted upon.
                    Fault("rejected an inbound message: " + exception.Message);
                }
            }
        }

        private void Fault(string detail)
        {
            _state = TransportState.Faulted;
            _statusDetail = detail;
            _faults.Enqueue(detail);
        }
    }
}
