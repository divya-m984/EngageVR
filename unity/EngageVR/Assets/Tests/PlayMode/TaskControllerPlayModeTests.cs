// EngageVR -- PlayMode tests for the TaskController MonoBehaviour.
//
// These run entirely against MockTransport: no server, no network, no
// headset. They check that the controller emits a well-formed protocol
// conversation and that an adaptation command is applied and answered.
//
// STATUS: not executed. No Unity Editor is installed in the environment
// where this code was written, so PlayMode validation remains pending.

using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EngageVR.Tests.PlayMode
{
    using EngageVR.Networking;
    using EngageVR.Protocol;
    using EngageVR.Task;

    [TestFixture]
    public sealed class TaskControllerPlayModeTests
    {
        private GameObject _host;
        private TaskController _controller;

        private MockTransport Transport
        {
            get { return (MockTransport)_controller.Transport; }
        }

        [SetUp]
        public void SetUp()
        {
            _host = new GameObject("TaskControllerUnderTest");
            _controller = _host.AddComponent<TaskController>();
            _controller.useMockTransport = true;
            _controller.sessionId = "playmode-session";
            _controller.blocks = 1;
            _controller.trialsPerBlock = 2;
            _controller.responseTimeoutMs = 50f;
            _controller.interTrialIntervalMs = 10f;
            _controller.feedbackDurationMs = 10f;
        }

        [TearDown]
        public void TearDown()
        {
            if (_host != null)
            {
                Object.DestroyImmediate(_host);
            }
        }

        [UnityTest]
        public IEnumerator HandshakeIsSentOnStart()
        {
            yield return null; // let Start run

            Assert.That(Transport.Sent.Count, Is.GreaterThanOrEqualTo(1));
            Assert.That(
                Transport.Sent[0].MessageTypeValue, Is.EqualTo(MessageType.ClientHello));
            Assert.That(Transport.Sent[0].SequenceNumber, Is.EqualTo(0));
            Assert.That(Transport.Sent[0].Source, Is.EqualTo(MessageSource.UnityClient));
        }

        [UnityTest]
        public IEnumerator SequenceNumbersNeverDecrease()
        {
            yield return null;
            yield return new WaitForSeconds(0.5f);

            int previous = -1;
            foreach (ProtocolEnvelope envelope in Transport.Sent)
            {
                Assert.That(envelope.SequenceNumber, Is.GreaterThan(previous));
                previous = envelope.SequenceNumber;
            }
        }

        [UnityTest]
        public IEnumerator EveryEmittedMessageIsValidAndCarriesTheSession()
        {
            yield return null;
            yield return new WaitForSeconds(0.3f);

            foreach (ProtocolEnvelope envelope in Transport.Sent)
            {
                Assert.That(
                    envelope.ProtocolVersionValue, Is.EqualTo(ProtocolVersion.Current));
                Assert.That(MessageType.IsKnown(envelope.MessageTypeValue), Is.True);
                Assert.That(envelope.SessionId, Is.EqualTo("playmode-session"));
                Assert.That(string.IsNullOrEmpty(envelope.MessageId), Is.False);
            }
        }

        [UnityTest]
        public IEnumerator AdaptationCommandIsAppliedAndAcknowledged()
        {
            yield return null;

            JsonValue payload = JsonValue.Object()
                .Set("command_id", JsonValue.String("pm-cmd-1"))
                .Set("command", JsonValue.String(AdaptationCommandName.SetDifficulty))
                .Set("value", JsonValue.Number(5))
                .Set("reason", JsonValue.String("playmode test"))
                .Set(
                    "issued_at_utc",
                    JsonValue.String(ProtocolEnvelope.FormatUtc(System.DateTime.UtcNow)))
                .Set("expires_at_utc", JsonValue.Null)
                .Set("target_role", JsonValue.String(ClientRole.Unity))
                .Set("target_client_id", JsonValue.Null)
                .Set("is_manual", JsonValue.Bool(true));

            Transport.InjectAdaptationCommand(payload, "playmode-session");
            yield return null;

            Assert.That(_controller.RuntimeState.DifficultyLevel, Is.EqualTo(5));

            bool acknowledged = false;
            foreach (ProtocolEnvelope envelope in Transport.Sent)
            {
                if (envelope.MessageTypeValue != MessageType.AdaptationAcknowledgement)
                {
                    continue;
                }

                acknowledged = true;
                Assert.That(
                    envelope.Payload.GetRequiredString("command_id"), Is.EqualTo("pm-cmd-1"));
                Assert.That(envelope.Payload.GetBoolOrNull("accepted"), Is.True);
                Assert.That(envelope.Payload.GetStringOrNull("applied_at_utc"), Is.Not.Null);
            }

            Assert.That(acknowledged, Is.True, "the command must be acknowledged");
        }

        [UnityTest]
        public IEnumerator NoMessageCarriesAnEngagementOrCognitiveLoadField()
        {
            yield return null;
            yield return new WaitForSeconds(0.3f);

            foreach (ProtocolEnvelope envelope in Transport.Sent)
            {
                string text = envelope.ToJsonString().ToLowerInvariant();
                Assert.That(text, Does.Not.Contain("engagement"));
                Assert.That(text, Does.Not.Contain("cognitive_load"));
                Assert.That(text, Does.Not.Contain("attention"));
                Assert.That(text, Does.Not.Contain("fatigue"));
            }
        }
    }
}
