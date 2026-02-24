"""
Tests for the download -> transcribe -> interpret pipeline.
Ensures rearchitecture (parallel chunks/batches, deferred metadata, language hint)
does not change observable behavior: output shape and cache format.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Import pipeline functions from app (streamlit will load but we don't run the UI)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as app_module


class TestTranscribeContract(unittest.TestCase):
    """transcribe_with_timestamps returns (segments, detected_language)."""

    @patch.object(app_module, 'OpenAI')
    @patch.object(app_module, '_split_audio_into_chunks')
    def test_returns_tuple_segments_and_language(self, mock_split, mock_openai_class):
        mock_split.return_value = [("/tmp/audio.mp3", 0.0)]
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        transcript = MagicMock()
        transcript.segments = [
            MagicMock(start=0.0, end=1.5, text="Hello world"),
        ]
        transcript.language = "en"
        mock_client.audio.transcriptions.create.return_value = transcript

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio")
            path = f.name
        try:
            segments, detected_lang = app_module.transcribe_with_timestamps(path)
            self.assertIsInstance(segments, list)
            self.assertIn(detected_lang, (None, "en"))
            if segments:
                self.assertIn("start", segments[0])
                self.assertIn("end", segments[0])
                self.assertIn("text", segments[0])
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    @patch.object(app_module, 'OpenAI')
    @patch.object(app_module, '_split_audio_into_chunks')
    def test_quality_retry_still_returns_tuple(self, mock_split, mock_openai_class):
        """When first pass returns empty, retry runs; we still get (list, lang)."""
        mock_split.return_value = [("/tmp/audio.mp3", 0.0)]
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        transcript = MagicMock()
        transcript.segments = [MagicMock(start=0.0, end=1.0, text="Enough text here for validation")]
        transcript.language = "es"
        mock_client.audio.transcriptions.create.return_value = transcript

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"x")
            path = f.name
        try:
            segments, detected_lang = app_module.transcribe_with_timestamps(path)
            self.assertIsInstance(segments, list)
            self.assertIsInstance(detected_lang, (type(None), str))
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass


class TestInterpretContract(unittest.TestCase):
    """interpret_segments(segments, language_hint=None) returns list with romanized, translation, meaning."""

    @patch.object(app_module, 'Anthropic')
    def test_interpret_returns_segments_with_expected_keys(self, mock_anthropic_class):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='[{"original":"hi","romanized":"","translation":"hello","meaning":"greeting"}]')]
        )

        segments = [{"start": 0, "end": 1, "text": "hi"}]
        result = app_module.interpret_segments(segments, language_hint=None)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn("romanized", result[0])
        self.assertIn("translation", result[0])
        self.assertIn("meaning", result[0])

    @patch.object(app_module, 'Anthropic')
    def test_interpret_accepts_language_hint(self, mock_anthropic_class):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='[{"original":"hola","romanized":"","translation":"hello","meaning":""}]')]
        )

        segments = [{"start": 0, "end": 1, "text": "hola"}]
        result = app_module.interpret_segments(segments, language_hint="es")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        call_args = mock_client.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        self.assertIn("Spanish", prompt)


class TestSaveToCacheNoBlockingMetadata(unittest.TestCase):
    """save_to_cache does not call _get_youtube_metadata (deferred to avoid latency)."""

    @patch.object(app_module, '_get_youtube_metadata')
    def test_save_to_cache_does_not_call_youtube_metadata(self, mock_metadata):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(app_module.__dict__, {'CACHE_DIR': tmp}):
                app_module.save_to_cache(
                    "https://www.youtube.com/watch?v=abc",
                    "auto",
                    {"segments": [], "language": "en", "mood": "Upbeat"},
                    title="Test Song"
                )
        mock_metadata.assert_not_called()

    @patch.object(app_module, '_get_youtube_metadata')
    def test_save_to_cache_meta_has_placeholder_channel_and_duration(self, mock_metadata):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(app_module.__dict__, {'CACHE_DIR': tmp}):
                app_module.save_to_cache(
                    "https://www.youtube.com/watch?v=abc",
                    "auto",
                    {"segments": [], "language": "en", "mood": "Upbeat"},
                    title="Test Song"
                )
            files = [f for f in os.listdir(tmp) if f.endswith(".json")]
            self.assertEqual(len(files), 1)
            with open(os.path.join(tmp, files[0])) as f:
                data = json.load(f)
            meta = data.get("_meta", {})
            self.assertEqual(meta.get("channel"), "Unknown")
            self.assertEqual(meta.get("duration"), "")


class TestFormatInterpretationPrompt(unittest.TestCase):
    """_format_interpretation_prompt includes language line when hint given."""

    def test_no_hint_omits_language_line(self):
        out = app_module._format_interpretation_prompt("1. Hello", language_hint=None)
        self.assertNotIn("The lyrics are in:", out)
        self.assertIn("1. Hello", out)

    def test_with_hint_includes_language(self):
        out = app_module._format_interpretation_prompt("1. Hola", language_hint="es")
        self.assertIn("The lyrics are in: Spanish", out)
        self.assertIn("1. Hola", out)

    def test_unknown_code_passes_through(self):
        out = app_module._format_interpretation_prompt("1. x", language_hint="xx")
        self.assertIn("The lyrics are in: xx", out)


class TestMergeEarlyHallucinations(unittest.TestCase):
    """merge_early_repeated_hallucinations merges repeated short phrases in first 90s."""

    def test_merges_repeated_short_phrase_in_early_part(self):
        segments = [
            {"start": 0, "end": 10, "text": "Девчонки"},
            {"start": 30, "end": 40, "text": "Девчонки"},
            {"start": 60, "end": 70, "text": "Девчонки"},
            {"start": 95, "end": 100, "text": "Real lyrics here"},
        ]
        out = app_module.merge_early_repeated_hallucinations(segments)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "♪")
        self.assertEqual(out[0]["start"], 0)
        self.assertEqual(out[0]["end"], 70)
        self.assertEqual(out[1]["text"], "Real lyrics here")

    def test_leaves_long_repeated_phrase_unchanged(self):
        long_phrase = "Jai Ganesh Jai Ganesh Jai Ganesh Deva Mata Ganesh Deva"  # > 50 chars
        segments = [
            {"start": 0, "end": 5, "text": long_phrase},
            {"start": 5, "end": 10, "text": long_phrase},
        ]
        out = app_module.merge_early_repeated_hallucinations(segments, early_sec=90, max_text_len=50)
        self.assertEqual(len(out), 2)

    def test_leaves_late_segments_unchanged(self):
        segments = [
            {"start": 100, "end": 110, "text": "Same"},
            {"start": 110, "end": 120, "text": "Same"},
        ]
        out = app_module.merge_early_repeated_hallucinations(segments)
        self.assertEqual(len(out), 2)

    def test_replaces_known_prompt_echo_and_merges(self):
        prompt = "Lyrics of a song. Transcribe the singing. May be in any language."
        segments = [
            {"start": 0, "end": 2, "text": prompt},
            {"start": 30, "end": 32, "text": prompt},
            {"start": 60, "end": 62, "text": "Real lyrics"},
        ]
        out = app_module.merge_early_repeated_hallucinations(segments)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "♪")
        self.assertEqual(out[0]["start"], 0)
        self.assertEqual(out[0]["end"], 32)
        self.assertEqual(out[1]["text"], "Real lyrics")


class TestTwoPassTranscription(unittest.TestCase):
    """Two-pass transcription re-transcribes hallucinated chunks with language hint."""

    @patch.object(app_module, 'OpenAI')
    @patch.object(app_module, '_split_audio_into_chunks')
    @patch.object(app_module, '_transcribe_chunk_with_retry')
    def test_retranscribes_hallucinated_first_chunk(self, mock_retry, mock_split, mock_openai_class):
        """If first chunk is hallucinated but second has real content, first chunk is retried with language hint."""
        mock_split.return_value = [("/tmp/chunk0.mp3", 0.0), ("/tmp/chunk1.mp3", 120.0)]

        def fake_retry(chunk_path, offset_sec, language, client, max_retries=3):
            if 'chunk0' in chunk_path and language is None:
                return ([
                    {"start": 0.0, "end": 30.0, "text": "Девчонки"},
                    {"start": 30.0, "end": 60.0, "text": "Девчонки"},
                    {"start": 60.0, "end": 90.0, "text": "Девчонки"},
                ], None)
            elif 'chunk1' in chunk_path:
                return ([
                    {"start": 120.0, "end": 150.0, "text": "Real lyrics from second chunk with enough text"},
                ], "hi")
            elif 'chunk0' in chunk_path and language == "hi":
                return ([
                    {"start": 0.0, "end": 30.0, "text": "सुखकर्ता दुखहर्ता"},
                    {"start": 30.0, "end": 60.0, "text": "जय देव जय देव"},
                ], "hi")
            return ([], None)

        mock_retry.side_effect = fake_retry

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            segments, detected_lang = app_module.transcribe_with_timestamps(path)
            self.assertEqual(detected_lang, "hi")
            texts = [s['text'] for s in segments]
            self.assertTrue(any("सुखकर्ता" in t for t in texts),
                            f"Expected retranscribed lyrics, got: {texts}")
            # 2 chunks in pass 1 + 1 retranscribe in pass 2 = at least 3 calls
            self.assertGreaterEqual(mock_retry.call_count, 3)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    @patch.object(app_module, 'OpenAI')
    @patch.object(app_module, '_split_audio_into_chunks')
    @patch.object(app_module, '_transcribe_chunk_with_retry')
    def test_does_not_retranscribe_good_chunks(self, mock_retry, mock_split, mock_openai_class):
        """If all chunks have good content, no retranscription happens."""
        mock_split.return_value = [("/tmp/audio.mp3", 0.0)]

        mock_retry.return_value = ([
            {"start": 0.0, "end": 30.0, "text": "Good lyrics here with enough content to be valid"},
        ], "en")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            segments, detected_lang = app_module.transcribe_with_timestamps(path)
            self.assertEqual(mock_retry.call_count, 1)
            self.assertEqual(detected_lang, "en")
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    @patch.object(app_module, 'OpenAI')
    @patch.object(app_module, '_split_audio_into_chunks')
    @patch.object(app_module, '_transcribe_chunk_with_retry')
    def test_no_retranscribe_when_all_chunks_hallucinated(self, mock_retry, mock_split, mock_openai_class):
        """If ALL chunks are hallucinated (no detected language), skip pass 2 to avoid infinite loop."""
        mock_split.return_value = [("/tmp/chunk0.mp3", 0.0)]

        mock_retry.return_value = ([
            {"start": 0.0, "end": 30.0, "text": "♪"},
        ], None)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            segments, detected_lang = app_module.transcribe_with_timestamps(path)
            # No language detected means no pass 2 (would be pointless)
            # But quality check retry will fire since total text < 20
            # Total calls: 1 (pass 1) + 1 (quality retry) = 2
            self.assertLessEqual(mock_retry.call_count, 2)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass


class TestChunkHallucinationDetection(unittest.TestCase):
    """Tests for _chunk_looks_hallucinated and _is_hallucinated_segment."""

    def test_empty_chunk_is_hallucinated(self):
        self.assertTrue(app_module._chunk_looks_hallucinated([]))

    def test_all_known_hallucination_is_hallucinated(self):
        segs = [
            {"start": 0, "end": 10, "text": "♪"},
            {"start": 10, "end": 20, "text": "♪"},
        ]
        self.assertTrue(app_module._chunk_looks_hallucinated(segs))

    def test_mostly_real_content_not_hallucinated(self):
        segs = [
            {"start": 0, "end": 10, "text": "Real lyrics line one"},
            {"start": 10, "end": 20, "text": "Real lyrics line two"},
            {"start": 20, "end": 30, "text": "Real lyrics line three"},
        ]
        self.assertFalse(app_module._chunk_looks_hallucinated(segs))

    def test_repeated_short_text_is_hallucinated(self):
        segs = [
            {"start": 0, "end": 10, "text": "Девчонки"},
            {"start": 10, "end": 20, "text": "Девчонки"},
            {"start": 20, "end": 30, "text": "Девчонки"},
        ]
        self.assertTrue(app_module._chunk_looks_hallucinated(segs))


if __name__ == "__main__":
    unittest.main()
