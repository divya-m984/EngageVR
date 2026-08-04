// EngageVR -- EditMode tests against the shared Python fixtures.
//
// These tests read the SAME files the Python test suite reads:
// <repo>/protocol/fixtures/. That is what makes "Unity and the
// simulator use the same protocol" a checked property rather than a
// claim. A field rename on either side fails the other side's tests.
//
// STATUS: these tests have NOT been executed. No Unity Editor is
// installed in the development environment where this code was
// written, so Unity compilation and test execution remain pending.

using System.Collections.Generic;
using System.IO;
using NUnit.Framework;

namespace EngageVR.Tests.EditMode
{
    using EngageVR.Protocol;
    using EngageVR.Task;

    /// <summary>Locates the shared fixture directory.</summary>
    public static class FixturePaths
    {
        /// <summary>
        /// The repository's protocol/fixtures directory.
        /// <para>
        /// Application.dataPath is &lt;repo&gt;/unity/EngageVR/Assets, so
        /// the fixtures are three levels up.
        /// </para>
        /// </summary>
        public static string Root
        {
            get
            {
                string assets = UnityEngine.Application.dataPath;
                string repository = Directory.GetParent(assets).Parent.Parent.FullName;
                return Path.Combine(repository, "protocol", "fixtures");
            }
        }

        public static string Valid(string name)
        {
            return Path.Combine(Root, "valid", name);
        }

        public static string Invalid(string name)
        {
            return Path.Combine(Root, "invalid", name);
        }

        public static bool Available
        {
            get { return Directory.Exists(Root); }
        }
    }

    [TestFixture]
    public sealed class JsonTests
    {
        [Test]
        public void NullIsDistinctFromEmptyStringAndZero()
        {
            JsonValue root = JsonValue.Object()
                .Set("absent_string", Json.OrNull((string)null))
                .Set("empty_string", JsonValue.String(string.Empty))
                .Set("absent_number", Json.OrNull((double?)null))
                .Set("zero_number", JsonValue.Number(0.0));

            string text = Json.Serialize(root);
            Assert.That(text, Does.Contain("\"absent_string\":null"));
            Assert.That(text, Does.Contain("\"empty_string\":\"\""));
            Assert.That(text, Does.Contain("\"absent_number\":null"));
            Assert.That(text, Does.Contain("\"zero_number\":0"));

            JsonValue parsed = Json.Parse(text);
            Assert.That(parsed.GetStringOrNull("absent_string"), Is.Null);
            Assert.That(parsed.GetStringOrNull("empty_string"), Is.EqualTo(string.Empty));
            Assert.That(parsed.GetDoubleOrNull("absent_number"), Is.Null);
            Assert.That(parsed.GetDoubleOrNull("zero_number"), Is.EqualTo(0.0));
        }

        [Test]
        public void RoundTripsNestedStructures()
        {
            const string text =
                "{\"a\":[1,2.5,null,true,\"x\"],\"b\":{\"c\":{\"d\":null}}}";
            JsonValue parsed = Json.Parse(text);
            Assert.That(parsed.Get("a").AsArray.Count, Is.EqualTo(5));
            Assert.That(parsed.Get("a").AsArray[2].IsNull, Is.True);
            Assert.That(parsed.Get("b").Get("c").Get("d").IsNull, Is.True);
        }

        [Test]
        public void RefusesToSerializeNaN()
        {
            JsonValue root = JsonValue.Object().Set("bad", JsonValue.Number(double.NaN));
            Assert.Throws<JsonException>(() => Json.Serialize(root));
        }

        [Test]
        public void EscapesControlCharacters()
        {
            JsonValue root = JsonValue.Object().Set("s", JsonValue.String("a\"b\\c\nd"));
            string text = Json.Serialize(root);
            Assert.That(Json.Parse(text).GetRequiredString("s"), Is.EqualTo("a\"b\\c\nd"));
        }
    }

    [TestFixture]
    public sealed class ProtocolVersionTests
    {
        [Test]
        public void CurrentVersionIsSupported()
        {
            Assert.That(ProtocolVersion.IsSupported(ProtocolVersion.Current), Is.True);
        }

        [Test]
        public void DifferentMajorVersionIsRejected()
        {
            Assert.That(ProtocolVersion.IsSupported("2.0"), Is.False);
        }

        [Test]
        public void NewerMinorVersionIsAccepted()
        {
            Assert.That(ProtocolVersion.IsSupported("1.7"), Is.True);
        }

        [Test]
        public void MalformedVersionIsRejected()
        {
            Assert.That(ProtocolVersion.IsSupported("one"), Is.False);
            Assert.That(ProtocolVersion.IsSupported("1"), Is.False);
            Assert.That(ProtocolVersion.IsSupported(""), Is.False);
        }
    }

    [TestFixture]
    public sealed class SharedFixtureTests
    {
        [SetUp]
        public void RequireFixtures()
        {
            if (!FixturePaths.Available)
            {
                Assert.Ignore(
                    "shared fixtures not found at " + FixturePaths.Root
                    + "; run 'uv run python scripts/generate_protocol_artifacts.py'");
            }
        }

        [Test]
        public void EveryValidFixtureParses()
        {
            string[] files = Directory.GetFiles(
                Path.Combine(FixturePaths.Root, "valid"), "*.json");
            Assert.That(files.Length, Is.GreaterThan(0));

            foreach (string file in files)
            {
                ProtocolEnvelope envelope = ProtocolEnvelope.Parse(File.ReadAllText(file));
                Assert.That(
                    envelope.ProtocolVersionValue,
                    Is.EqualTo(ProtocolVersion.Current),
                    file);
                Assert.That(MessageType.IsKnown(envelope.MessageTypeValue), Is.True, file);
                Assert.That(envelope.SequenceNumber, Is.GreaterThanOrEqualTo(0), file);
            }
        }

        [Test]
        public void UnsupportedMajorVersionFixtureIsRejected()
        {
            string text = File.ReadAllText(
                FixturePaths.Invalid("unsupported-major-version.json"));
            Assert.Throws<JsonException>(() => ProtocolEnvelope.Parse(text));
        }

        [Test]
        public void UnknownMessageTypeFixtureIsRejected()
        {
            string text = File.ReadAllText(FixturePaths.Invalid("unknown-message-type.json"));
            Assert.Throws<JsonException>(() => ProtocolEnvelope.Parse(text));
        }

        [Test]
        public void TimeoutFixtureCarriesNoReactionTimeOrResponse()
        {
            ProtocolEnvelope envelope = ProtocolEnvelope.Parse(
                File.ReadAllText(FixturePaths.Valid("task-event-response-timeout.json")));
            TaskEventDetail detail = TaskEventDetail.FromPayload(envelope.Payload);

            Assert.That(detail.EventType, Is.EqualTo(TaskEventType.ResponseTimeout));
            Assert.That(detail.ResponseOutcomeValue, Is.EqualTo(ResponseOutcome.Timeout));
            Assert.That(detail.ReactionTimeMs, Is.Null, "a timeout has no reaction time");
            Assert.That(detail.ObservedResponse, Is.Null, "a timeout has no response");
            Assert.That(detail.ResponseCorrect, Is.Null, "a timeout has no correctness");
        }

        [Test]
        public void ResponseFixtureCarriesReactionTimeAndCorrectness()
        {
            ProtocolEnvelope envelope = ProtocolEnvelope.Parse(
                File.ReadAllText(FixturePaths.Valid("task-event-response-registered.json")));
            TaskEventDetail detail = TaskEventDetail.FromPayload(envelope.Payload);

            Assert.That(detail.ReactionTimeMs, Is.EqualTo(412.5));
            Assert.That(detail.ResponseCorrect, Is.True);
            Assert.That(detail.ObservedResponse, Is.EqualTo("j"));
        }

        [Test]
        public void ReplayedSyntheticFixtureCarriesBothLabels()
        {
            ProtocolEnvelope envelope = ProtocolEnvelope.Parse(
                File.ReadAllText(FixturePaths.Valid("replayed-synthetic-task-event.json")));

            Assert.That(
                envelope.Provenance.SyntheticLabel,
                Is.EqualTo(DataSource.SyntheticLabel),
                "a replayed synthetic message stays SYNTHETIC");
            Assert.That(envelope.Replay, Is.Not.Null);
            Assert.That(envelope.Replay.ReplayLabel, Is.EqualTo(DataSource.ReplayLabel));
            Assert.That(envelope.Replay.SourceSessionId, Is.EqualTo("recorded-session"));
        }

        [Test]
        public void LiveFixtureHasNoSyntheticLabelAndNoReplayBlock()
        {
            ProtocolEnvelope envelope = ProtocolEnvelope.Parse(
                File.ReadAllText(FixturePaths.Valid("server-hello.json")));
            Assert.That(envelope.Provenance.SyntheticLabel, Is.Null);
            Assert.That(envelope.Replay, Is.Null);
        }

        [Test]
        public void RoundTripPreservesEveryFixtureField()
        {
            string[] files = Directory.GetFiles(
                Path.Combine(FixturePaths.Root, "valid"), "*.json");

            foreach (string file in files)
            {
                ProtocolEnvelope original = ProtocolEnvelope.Parse(File.ReadAllText(file));
                ProtocolEnvelope reparsed = ProtocolEnvelope.Parse(original.ToJsonString());

                Assert.That(reparsed.MessageId, Is.EqualTo(original.MessageId), file);
                Assert.That(reparsed.MessageTypeValue, Is.EqualTo(original.MessageTypeValue), file);
                Assert.That(reparsed.SessionId, Is.EqualTo(original.SessionId), file);
                Assert.That(reparsed.Source, Is.EqualTo(original.Source), file);
                Assert.That(reparsed.SequenceNumber, Is.EqualTo(original.SequenceNumber), file);
                Assert.That(reparsed.SentAtUtc, Is.EqualTo(original.SentAtUtc), file);
                Assert.That(
                    reparsed.SentAtMonotonicSeconds,
                    Is.EqualTo(original.SentAtMonotonicSeconds),
                    file);
                Assert.That(reparsed.CorrelationId, Is.EqualTo(original.CorrelationId), file);
                Assert.That(
                    reparsed.Provenance.SyntheticLabel,
                    Is.EqualTo(original.Provenance.SyntheticLabel),
                    file);
                Assert.That(
                    reparsed.Replay == null,
                    Is.EqualTo(original.Replay == null),
                    file);
            }
        }

        [Test]
        public void GeneratedTaskEventMatchesTheFixtureFieldSet()
        {
            // Build the same event the fixture holds and compare key sets,
            // so a field added on the Python side without a C# counterpart
            // is caught here rather than at runtime against a live backend.
            ProtocolEnvelope fixture = ProtocolEnvelope.Parse(
                File.ReadAllText(FixturePaths.Valid("task-event-stimulus-presented.json")));

            TaskEventDetail generated = new TaskEventDetail
            {
                EventType = TaskEventType.StimulusPresented,
                TaskId = "reaction_task_v1",
                BlockId = 0,
                TrialId = 3,
                StimulusId = "square-b0t3",
                StimulusCategory = "square",
                ExpectedResponse = "j",
                DifficultyLevel = 1,
                TaskElapsedMs = 4200.0,
                TrialElapsedMs = 500.0
            };

            Dictionary<string, JsonValue> fixtureEvent = fixture.Payload.Get("event").AsObject;
            Dictionary<string, JsonValue> generatedEvent = generated.ToJson().AsObject;

            CollectionAssert.AreEquivalent(
                new List<string>(fixtureEvent.Keys),
                new List<string>(generatedEvent.Keys),
                "the C# task-event field set must match the Python one exactly");

            foreach (KeyValuePair<string, JsonValue> pair in fixtureEvent)
            {
                Assert.That(
                    Json.Serialize(generatedEvent[pair.Key]),
                    Is.EqualTo(Json.Serialize(pair.Value)),
                    "field " + pair.Key);
            }
        }
    }
}
