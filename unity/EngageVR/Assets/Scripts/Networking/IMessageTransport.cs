// EngageVR -- transport abstraction for the Unity client.
//
// The task controller is written against this interface only, so the
// same task runs against a live backend or fully offline against
// MockTransport. That is what lets the Unity task be developed and
// demonstrated with no Python process running.

using System;

namespace EngageVR.Networking
{
    using EngageVR.Protocol;

    /// <summary>Connection lifecycle states surfaced to the UI.</summary>
    public enum TransportState
    {
        Disconnected,
        Connecting,
        Connected,
        Closing,
        Faulted
    }

    /// <summary>
    /// Moves protocol envelopes between this client and a peer.
    /// <para>
    /// Implementations must be safe to poll from Unity's main thread:
    /// <see cref="Poll"/> is called from Update and is where queued
    /// inbound messages are dispatched, so handlers never run on a
    /// background thread.
    /// </para>
    /// </summary>
    public interface IMessageTransport : IDisposable
    {
        TransportState State { get; }

        /// <summary>Human-readable detail for the HUD, e.g. a fault reason.</summary>
        string StatusDetail { get; }

        /// <summary>Raised on the main thread for each inbound message.</summary>
        event Action<ProtocolEnvelope> MessageReceived;

        /// <summary>Raised on the main thread when the transport faults.</summary>
        event Action<string> Faulted;

        /// <summary>Begin connecting. Returns immediately.</summary>
        void Connect();

        /// <summary>Queue one envelope for sending.</summary>
        void Send(ProtocolEnvelope envelope);

        /// <summary>
        /// Dispatch queued inbound messages. Call once per frame from
        /// Update.
        /// </summary>
        void Poll();

        /// <summary>Close the transport. Idempotent.</summary>
        void Close();
    }
}
