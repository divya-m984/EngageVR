// EngageVR -- minimal dependency-free JSON reader/writer.
//
// WHY NOT UnityEngine.JsonUtility
// -------------------------------
// JsonUtility cannot represent null. It serializes a null string field
// as "" and a nullable numeric field as 0, and on deserialization it
// leaves absent fields at their default value. The EngageVR protocol
// depends on the difference between "no response" (null) and "a
// response of 0 ms", and between "no reaction time" and "0 ms". Using
// JsonUtility would silently convert every missing response into a
// zero-latency response, which is exactly the failure the Python
// schema forbids. JsonUtility also cannot serialize dictionaries or
// top-level arrays, both of which the protocol uses.
//
// WHY NOT Newtonsoft.Json
// -----------------------
// com.unity.nuget.newtonsoft-json would work, but adding a package
// dependency that has not been resolved or compiled in this repository
// would be an unverified claim. This file has no dependencies beyond
// the .NET base class library, so it compiles wherever C# does.
//
// Scope: this is a protocol serializer, not a general JSON library. It
// implements exactly what the EngageVR protocol needs.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace EngageVR.Protocol
{
    /// <summary>The kind of value a <see cref="JsonValue"/> holds.</summary>
    public enum JsonKind
    {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object
    }

    /// <summary>
    /// A parsed JSON value. Null is a first-class kind, distinct from an
    /// absent key and from a zero or empty-string value.
    /// </summary>
    public sealed class JsonValue
    {
        public JsonKind Kind { get; private set; }

        private bool _bool;
        private double _number;
        private string _string;
        private List<JsonValue> _array;
        private Dictionary<string, JsonValue> _object;

        public static readonly JsonValue Null = new JsonValue { Kind = JsonKind.Null };

        public static JsonValue Bool(bool value)
        {
            return new JsonValue { Kind = JsonKind.Bool, _bool = value };
        }

        public static JsonValue Number(double value)
        {
            return new JsonValue { Kind = JsonKind.Number, _number = value };
        }

        public static JsonValue String(string value)
        {
            if (value == null)
            {
                return Null;
            }

            return new JsonValue { Kind = JsonKind.String, _string = value };
        }

        public static JsonValue Array(List<JsonValue> items)
        {
            return new JsonValue
            {
                Kind = JsonKind.Array,
                _array = items ?? new List<JsonValue>()
            };
        }

        public static JsonValue Object()
        {
            return new JsonValue
            {
                Kind = JsonKind.Object,
                _object = new Dictionary<string, JsonValue>(StringComparer.Ordinal)
            };
        }

        public bool IsNull
        {
            get { return Kind == JsonKind.Null; }
        }

        public bool AsBool
        {
            get
            {
                Require(JsonKind.Bool);
                return _bool;
            }
        }

        public double AsNumber
        {
            get
            {
                Require(JsonKind.Number);
                return _number;
            }
        }

        public string AsString
        {
            get
            {
                Require(JsonKind.String);
                return _string;
            }
        }

        public List<JsonValue> AsArray
        {
            get
            {
                Require(JsonKind.Array);
                return _array;
            }
        }

        public Dictionary<string, JsonValue> AsObject
        {
            get
            {
                Require(JsonKind.Object);
                return _object;
            }
        }

        /// <summary>Set a member. Only valid on an object.</summary>
        public JsonValue Set(string key, JsonValue value)
        {
            Require(JsonKind.Object);
            _object[key] = value ?? Null;
            return this;
        }

        /// <summary>Set a member only when the value is not null.</summary>
        public JsonValue SetIfPresent(string key, JsonValue value)
        {
            if (value != null && !value.IsNull)
            {
                Set(key, value);
            }
            else
            {
                Set(key, Null);
            }

            return this;
        }

        public bool Has(string key)
        {
            return Kind == JsonKind.Object && _object.ContainsKey(key);
        }

        /// <summary>
        /// Get a member, or <see cref="Null"/> when the key is absent.
        /// Absent and explicitly-null are deliberately treated the same
        /// by readers: both mean "no value was supplied".
        /// </summary>
        public JsonValue Get(string key)
        {
            if (Kind != JsonKind.Object)
            {
                return Null;
            }

            JsonValue found;
            return _object.TryGetValue(key, out found) ? found : Null;
        }

        public string GetStringOrNull(string key)
        {
            JsonValue value = Get(key);
            return value.Kind == JsonKind.String ? value._string : null;
        }

        public string GetRequiredString(string key)
        {
            JsonValue value = Get(key);
            if (value.Kind != JsonKind.String)
            {
                throw new JsonException("missing required string field '" + key + "'");
            }

            return value._string;
        }

        public double? GetDoubleOrNull(string key)
        {
            JsonValue value = Get(key);
            return value.Kind == JsonKind.Number ? (double?)value._number : null;
        }

        public double GetRequiredDouble(string key)
        {
            JsonValue value = Get(key);
            if (value.Kind != JsonKind.Number)
            {
                throw new JsonException("missing required number field '" + key + "'");
            }

            return value._number;
        }

        public int? GetIntOrNull(string key)
        {
            double? value = GetDoubleOrNull(key);
            return value.HasValue ? (int?)Convert.ToInt32(value.Value) : null;
        }

        public int GetRequiredInt(string key)
        {
            return Convert.ToInt32(GetRequiredDouble(key));
        }

        public bool? GetBoolOrNull(string key)
        {
            JsonValue value = Get(key);
            return value.Kind == JsonKind.Bool ? (bool?)value._bool : null;
        }

        private void Require(JsonKind kind)
        {
            if (Kind != kind)
            {
                throw new JsonException(
                    "expected a JSON " + kind + " but this value is a " + Kind);
            }
        }

        public override string ToString()
        {
            return Json.Serialize(this);
        }
    }

    /// <summary>A JSON parsing or type error.</summary>
    public sealed class JsonException : Exception
    {
        public JsonException(string message) : base(message)
        {
        }
    }

    /// <summary>JSON text to <see cref="JsonValue"/> and back.</summary>
    public static class Json
    {
        // Optional helpers, so callers can express "null" explicitly.

        public static JsonValue OrNull(string value)
        {
            return value == null ? JsonValue.Null : JsonValue.String(value);
        }

        public static JsonValue OrNull(double? value)
        {
            return value.HasValue ? JsonValue.Number(value.Value) : JsonValue.Null;
        }

        public static JsonValue OrNull(int? value)
        {
            return value.HasValue ? JsonValue.Number(value.Value) : JsonValue.Null;
        }

        public static JsonValue OrNull(bool? value)
        {
            return value.HasValue ? JsonValue.Bool(value.Value) : JsonValue.Null;
        }

        public static string Serialize(JsonValue value)
        {
            StringBuilder builder = new StringBuilder(256);
            Write(builder, value ?? JsonValue.Null);
            return builder.ToString();
        }

        public static JsonValue Parse(string text)
        {
            if (text == null)
            {
                throw new JsonException("cannot parse null text");
            }

            int index = 0;
            JsonValue value = ParseValue(text, ref index);
            SkipWhitespace(text, ref index);
            if (index != text.Length)
            {
                throw new JsonException(
                    "trailing characters after the JSON value at offset " + index);
            }

            return value;
        }

        // -- writing ------------------------------------------------------

        private static void Write(StringBuilder builder, JsonValue value)
        {
            switch (value.Kind)
            {
                case JsonKind.Null:
                    builder.Append("null");
                    break;
                case JsonKind.Bool:
                    builder.Append(value.AsBool ? "true" : "false");
                    break;
                case JsonKind.Number:
                    WriteNumber(builder, value.AsNumber);
                    break;
                case JsonKind.String:
                    WriteString(builder, value.AsString);
                    break;
                case JsonKind.Array:
                    WriteArray(builder, value);
                    break;
                case JsonKind.Object:
                    WriteObject(builder, value);
                    break;
                default:
                    throw new JsonException("unhandled JSON kind " + value.Kind);
            }
        }

        private static void WriteNumber(StringBuilder builder, double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                // JSON has no representation for these, and emitting a
                // placeholder would fabricate a value. Refuse instead.
                throw new JsonException(
                    "JSON cannot represent NaN or Infinity; refusing to substitute a value");
            }

            if (value == Math.Floor(value) && Math.Abs(value) < 1e15)
            {
                builder.Append(((long)value).ToString(CultureInfo.InvariantCulture));
                return;
            }

            builder.Append(value.ToString("R", CultureInfo.InvariantCulture));
        }

        private static void WriteString(StringBuilder builder, string value)
        {
            builder.Append('"');
            foreach (char c in value)
            {
                switch (c)
                {
                    case '"':
                        builder.Append("\\\"");
                        break;
                    case '\\':
                        builder.Append("\\\\");
                        break;
                    case '\b':
                        builder.Append("\\b");
                        break;
                    case '\f':
                        builder.Append("\\f");
                        break;
                    case '\n':
                        builder.Append("\\n");
                        break;
                    case '\r':
                        builder.Append("\\r");
                        break;
                    case '\t':
                        builder.Append("\\t");
                        break;
                    default:
                        if (c < ' ')
                        {
                            builder.Append("\\u");
                            builder.Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        }
                        else
                        {
                            builder.Append(c);
                        }

                        break;
                }
            }

            builder.Append('"');
        }

        private static void WriteArray(StringBuilder builder, JsonValue value)
        {
            builder.Append('[');
            List<JsonValue> items = value.AsArray;
            for (int i = 0; i < items.Count; i++)
            {
                if (i > 0)
                {
                    builder.Append(',');
                }

                Write(builder, items[i]);
            }

            builder.Append(']');
        }

        private static void WriteObject(StringBuilder builder, JsonValue value)
        {
            builder.Append('{');
            bool first = true;
            foreach (KeyValuePair<string, JsonValue> pair in value.AsObject)
            {
                if (!first)
                {
                    builder.Append(',');
                }

                first = false;
                WriteString(builder, pair.Key);
                builder.Append(':');
                Write(builder, pair.Value);
            }

            builder.Append('}');
        }

        // -- parsing ------------------------------------------------------

        private static void SkipWhitespace(string text, ref int index)
        {
            while (index < text.Length)
            {
                char c = text[index];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                {
                    index++;
                }
                else
                {
                    return;
                }
            }
        }

        private static JsonValue ParseValue(string text, ref int index)
        {
            SkipWhitespace(text, ref index);
            if (index >= text.Length)
            {
                throw new JsonException("unexpected end of JSON input");
            }

            char c = text[index];
            switch (c)
            {
                case '{':
                    return ParseObject(text, ref index);
                case '[':
                    return ParseArray(text, ref index);
                case '"':
                    return JsonValue.String(ParseString(text, ref index));
                case 't':
                    Expect(text, ref index, "true");
                    return JsonValue.Bool(true);
                case 'f':
                    Expect(text, ref index, "false");
                    return JsonValue.Bool(false);
                case 'n':
                    Expect(text, ref index, "null");
                    return JsonValue.Null;
                default:
                    return JsonValue.Number(ParseNumber(text, ref index));
            }
        }

        private static void Expect(string text, ref int index, string literal)
        {
            if (index + literal.Length > text.Length ||
                string.CompareOrdinal(text, index, literal, 0, literal.Length) != 0)
            {
                throw new JsonException(
                    "expected '" + literal + "' at offset " + index);
            }

            index += literal.Length;
        }

        private static JsonValue ParseObject(string text, ref int index)
        {
            index++; // '{'
            JsonValue result = JsonValue.Object();
            SkipWhitespace(text, ref index);
            if (index < text.Length && text[index] == '}')
            {
                index++;
                return result;
            }

            while (true)
            {
                SkipWhitespace(text, ref index);
                if (index >= text.Length || text[index] != '"')
                {
                    throw new JsonException("expected an object key at offset " + index);
                }

                string key = ParseString(text, ref index);
                SkipWhitespace(text, ref index);
                if (index >= text.Length || text[index] != ':')
                {
                    throw new JsonException("expected ':' at offset " + index);
                }

                index++;
                result.Set(key, ParseValue(text, ref index));
                SkipWhitespace(text, ref index);
                if (index >= text.Length)
                {
                    throw new JsonException("unterminated JSON object");
                }

                if (text[index] == ',')
                {
                    index++;
                    continue;
                }

                if (text[index] == '}')
                {
                    index++;
                    return result;
                }

                throw new JsonException("expected ',' or '}' at offset " + index);
            }
        }

        private static JsonValue ParseArray(string text, ref int index)
        {
            index++; // '['
            List<JsonValue> items = new List<JsonValue>();
            SkipWhitespace(text, ref index);
            if (index < text.Length && text[index] == ']')
            {
                index++;
                return JsonValue.Array(items);
            }

            while (true)
            {
                items.Add(ParseValue(text, ref index));
                SkipWhitespace(text, ref index);
                if (index >= text.Length)
                {
                    throw new JsonException("unterminated JSON array");
                }

                if (text[index] == ',')
                {
                    index++;
                    continue;
                }

                if (text[index] == ']')
                {
                    index++;
                    return JsonValue.Array(items);
                }

                throw new JsonException("expected ',' or ']' at offset " + index);
            }
        }

        private static string ParseString(string text, ref int index)
        {
            index++; // opening quote
            StringBuilder builder = new StringBuilder();
            while (true)
            {
                if (index >= text.Length)
                {
                    throw new JsonException("unterminated JSON string");
                }

                char c = text[index++];
                if (c == '"')
                {
                    return builder.ToString();
                }

                if (c != '\\')
                {
                    builder.Append(c);
                    continue;
                }

                if (index >= text.Length)
                {
                    throw new JsonException("unterminated escape sequence");
                }

                char escape = text[index++];
                switch (escape)
                {
                    case '"':
                        builder.Append('"');
                        break;
                    case '\\':
                        builder.Append('\\');
                        break;
                    case '/':
                        builder.Append('/');
                        break;
                    case 'b':
                        builder.Append('\b');
                        break;
                    case 'f':
                        builder.Append('\f');
                        break;
                    case 'n':
                        builder.Append('\n');
                        break;
                    case 'r':
                        builder.Append('\r');
                        break;
                    case 't':
                        builder.Append('\t');
                        break;
                    case 'u':
                        if (index + 4 > text.Length)
                        {
                            throw new JsonException("truncated \\u escape");
                        }

                        builder.Append((char)Convert.ToInt32(text.Substring(index, 4), 16));
                        index += 4;
                        break;
                    default:
                        throw new JsonException("unknown escape '\\" + escape + "'");
                }
            }
        }

        private static double ParseNumber(string text, ref int index)
        {
            int start = index;
            if (index < text.Length && (text[index] == '-' || text[index] == '+'))
            {
                index++;
            }

            while (index < text.Length)
            {
                char c = text[index];
                if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                    c == '+' || c == '-')
                {
                    index++;
                }
                else
                {
                    break;
                }
            }

            string slice = text.Substring(start, index - start);
            double parsed;
            if (!double.TryParse(
                    slice,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out parsed))
            {
                throw new JsonException("invalid number '" + slice + "' at offset " + start);
            }

            return parsed;
        }
    }
}
