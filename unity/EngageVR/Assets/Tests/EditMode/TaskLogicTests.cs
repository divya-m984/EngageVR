// EngageVR -- EditMode tests for the trial loop and adaptation handling.
//
// The classes under test take no UnityEngine dependency and read their
// clock from a parameter, so a whole task runs here without entering
// play mode and without real time passing.
//
// STATUS: not executed. No Unity Editor is installed in the environment
// where this code was written.

using System;
using NUnit.Framework;

namespace EngageVR.Tests.EditMode
{
    using EngageVR.Protocol;
    using EngageVR.Task;

    [TestFixture]
    public sealed class TrialPlanTests
    {
        [Test]
        public void PlanHasTheRequestedGeometry()
        {
            var plan = TrialPlanBuilder.Build(3, 7, 42);
            Assert.That(plan.Count, Is.EqualTo(21));
            Assert.That(plan[0].BlockId, Is.EqualTo(0));
            Assert.That(plan[20].BlockId, Is.EqualTo(2));
            Assert.That(plan[20].TrialId, Is.EqualTo(6));
        }

        [Test]
        public void SameSeedGivesTheSamePlan()
        {
            var first = TrialPlanBuilder.Build(2, 10, 42);
            var second = TrialPlanBuilder.Build(2, 10, 42);
            for (int i = 0; i < first.Count; i++)
            {
                Assert.That(second[i].StimulusId, Is.EqualTo(first[i].StimulusId));
                Assert.That(second[i].ExpectedResponse, Is.EqualTo(first[i].ExpectedResponse));
            }
        }

        [Test]
        public void EveryTrialHasAResponseKeyFromTheVocabulary()
        {
            foreach (var trial in TrialPlanBuilder.Build(2, 10, 7))
            {
                Assert.That(TrialPlanBuilder.ResponseKeys, Contains.Item(trial.ExpectedResponse));
                Assert.That(
                    TrialPlanBuilder.StimulusCategories, Contains.Item(trial.StimulusCategory));
            }
        }

        [Test]
        public void InvalidGeometryIsRejected()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => TrialPlanBuilder.Build(0, 5, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => TrialPlanBuilder.Build(1, 0, 1));
        }
    }

    [TestFixture]
    public sealed class TrialControllerTests
    {
        private static TrialDefinition Definition()
        {
            return new TrialDefinition
            {
                BlockId = 0,
                TrialId = 0,
                StimulusId = "square-b0t0",
                StimulusCategory = "square",
                ExpectedResponse = "j"
            };
        }

        [Test]
        public void CorrectResponseIsRecordedWithItsReactionTime()
        {
            TrialController trial = new TrialController(1500.0);
            trial.Begin(Definition(), 1000.0);
            Assert.That(trial.RegisterResponse("j", 1420.0), Is.True);

            Assert.That(trial.Resolution, Is.EqualTo(TrialResolution.Responded));
            Assert.That(trial.ReactionTimeMs, Is.EqualTo(420.0));
            Assert.That(trial.ResponseCorrect, Is.True);
        }

        [Test]
        public void IncorrectResponseIsStillAResponse()
        {
            TrialController trial = new TrialController(1500.0);
            trial.Begin(Definition(), 0.0);
            trial.RegisterResponse("k", 300.0);

            Assert.That(trial.Resolution, Is.EqualTo(TrialResolution.Responded));
            Assert.That(trial.ResponseCorrect, Is.False);
            Assert.That(trial.ReactionTimeMs, Is.EqualTo(300.0));
        }

        [Test]
        public void TimeoutLeavesResponseFieldsNull()
        {
            TrialController trial = new TrialController(1500.0);
            trial.Begin(Definition(), 0.0);

            Assert.That(trial.Tick(1499.0), Is.False);
            Assert.That(trial.Tick(1500.0), Is.True);

            Assert.That(trial.Resolution, Is.EqualTo(TrialResolution.TimedOut));
            Assert.That(trial.ReactionTimeMs, Is.Null, "a timeout is not a 0 ms response");
            Assert.That(trial.ObservedResponse, Is.Null);
            Assert.That(trial.ResponseCorrect, Is.Null);
        }

        [Test]
        public void ResponseAfterResolutionIsIgnored()
        {
            TrialController trial = new TrialController(1000.0);
            trial.Begin(Definition(), 0.0);
            trial.Tick(1000.0);

            Assert.That(trial.RegisterResponse("j", 1100.0), Is.False);
            Assert.That(trial.Resolution, Is.EqualTo(TrialResolution.TimedOut));
        }

        [Test]
        public void NegativeReactionTimeIsRefusedRatherThanClamped()
        {
            TrialController trial = new TrialController(1000.0);
            trial.Begin(Definition(), 500.0);
            Assert.That(trial.RegisterResponse("j", 400.0), Is.False);
            Assert.That(trial.ReactionTimeMs, Is.Null);
        }
    }

    [TestFixture]
    public sealed class TaskTelemetryTests
    {
        private static TaskTelemetry Telemetry()
        {
            EnvelopeFactory factory = new EnvelopeFactory(
                "unity-session",
                MessageSource.UnityClient,
                MessageProvenance.Live("test"),
                () => 0.0);
            return new TaskTelemetry(factory, "reaction_task_v1");
        }

        [Test]
        public void SequenceNumbersIncreaseByOne()
        {
            TaskTelemetry telemetry = Telemetry();
            ProtocolEnvelope first = telemetry.ClientHello("c", "1");
            ProtocolEnvelope second = telemetry.SessionEnd(true, "done");

            Assert.That(first.SequenceNumber, Is.EqualTo(0));
            Assert.That(second.SequenceNumber, Is.EqualTo(1));
        }

        [Test]
        public void MessageIdsAreUnique()
        {
            TaskTelemetry telemetry = Telemetry();
            Assert.That(
                telemetry.ClientHello("c", "1").MessageId,
                Is.Not.EqualTo(telemetry.SessionEnd(true, "done").MessageId));
        }

        [Test]
        public void TimeoutEventSerializesNullsNotZeros()
        {
            TaskTelemetry telemetry = Telemetry();
            TaskEventDetail detail = telemetry.ResponseTimeout(
                0, 1, "square-b0t1", "square", "j", 900.0, 1500.0, 1);

            string text = Json.Serialize(detail.ToJson());
            Assert.That(text, Does.Contain("\"reaction_time_ms\":null"));
            Assert.That(text, Does.Contain("\"observed_response\":null"));
            Assert.That(text, Does.Contain("\"response_correct\":null"));
            Assert.That(text, Does.Contain("\"response_outcome\":\"timeout\""));
        }

        [Test]
        public void NegativeReactionTimeIsRejectedAtBuildTime()
        {
            TaskTelemetry telemetry = Telemetry();
            Assert.Throws<ArgumentOutOfRangeException>(
                () => telemetry.ResponseRegistered(
                    0, 0, "s", "square", "j", "j", true, -1.0, 0.0, 0.0, 1));
        }
    }

    [TestFixture]
    public sealed class AdaptationReceiverTests
    {
        private TaskRuntimeState _state;
        private AdaptationReceiver _receiver;

        [SetUp]
        public void SetUp()
        {
            _state = new TaskRuntimeState { State = TaskStateName.Running };
            _receiver = new AdaptationReceiver(_state);
        }

        private static AdaptationCommandPayload Command(
            string id, string name, JsonValue value, string expiresAtUtc)
        {
            return new AdaptationCommandPayload
            {
                CommandId = id,
                Command = name,
                Value = value ?? JsonValue.Null,
                Reason = "test",
                IssuedAtUtc = ProtocolEnvelope.FormatUtc(DateTime.UtcNow),
                ExpiresAtUtc = expiresAtUtc,
                TargetRole = ClientRole.Unity
            };
        }

        [Test]
        public void SetDifficultyIsApplied()
        {
            var ack = _receiver.Apply(
                Command("c1", AdaptationCommandName.SetDifficulty, JsonValue.Number(4), null),
                DateTime.UtcNow);

            Assert.That(ack.Accepted, Is.True);
            Assert.That(ack.AppliedAtUtc, Is.Not.Null);
            Assert.That(_state.DifficultyLevel, Is.EqualTo(4));
        }

        [Test]
        public void DuplicateCommandIdIsAbsorbedIdempotently()
        {
            var command = Command(
                "c1", AdaptationCommandName.SetDifficulty, JsonValue.Number(4), null);
            _receiver.Apply(command, DateTime.UtcNow);
            _state.DifficultyLevel = 99; // prove the repeat does not re-apply

            var repeat = _receiver.Apply(command, DateTime.UtcNow);

            Assert.That(repeat.Accepted, Is.True);
            Assert.That(repeat.Duplicate, Is.True);
            Assert.That(_state.DifficultyLevel, Is.EqualTo(99));
            Assert.That(_receiver.AppliedCount, Is.EqualTo(1));
        }

        [Test]
        public void ExpiredCommandIsRejectedWithAReason()
        {
            DateTime past = DateTime.UtcNow.AddMinutes(-5);
            var ack = _receiver.Apply(
                Command(
                    "c2",
                    AdaptationCommandName.SetDifficulty,
                    JsonValue.Number(2),
                    ProtocolEnvelope.FormatUtc(past)),
                DateTime.UtcNow);

            Assert.That(ack.Accepted, Is.False);
            Assert.That(ack.RejectionReason, Does.Contain("expired"));
            Assert.That(ack.AppliedAtUtc, Is.Null);
        }

        [Test]
        public void PauseRequiresRunning()
        {
            _state.State = TaskStateName.Idle;
            var ack = _receiver.Apply(
                Command("c3", AdaptationCommandName.PauseTask, null, null), DateTime.UtcNow);

            Assert.That(ack.Accepted, Is.False);
            Assert.That(ack.RejectionReason, Does.Contain("running"));
        }

        [Test]
        public void PauseThenResumeWorks()
        {
            var pause = _receiver.Apply(
                Command("c4", AdaptationCommandName.PauseTask, null, null), DateTime.UtcNow);
            Assert.That(pause.Accepted, Is.True);
            Assert.That(_state.State, Is.EqualTo(TaskStateName.Paused));

            var resume = _receiver.Apply(
                Command("c5", AdaptationCommandName.ResumeTask, null, null), DateTime.UtcNow);
            Assert.That(resume.Accepted, Is.True);
            Assert.That(_state.State, Is.EqualTo(TaskStateName.Running));
        }

        [Test]
        public void UnknownCommandIsRejected()
        {
            var ack = _receiver.Apply(
                Command("c6", "launch_rocket", null, null), DateTime.UtcNow);
            Assert.That(ack.Accepted, Is.False);
            Assert.That(ack.RejectionReason, Does.Contain("set_difficulty"));
        }

        [Test]
        public void NonPositiveStimulusIntervalIsRejected()
        {
            var ack = _receiver.Apply(
                Command(
                    "c7",
                    AdaptationCommandName.SetStimulusInterval,
                    JsonValue.Number(0),
                    null),
                DateTime.UtcNow);
            Assert.That(ack.Accepted, Is.False);
            Assert.That(ack.RejectionReason, Does.Contain("positive"));
        }
    }
}
