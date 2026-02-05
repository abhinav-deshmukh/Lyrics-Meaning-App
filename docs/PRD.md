# Product requirements document

## Problem statement

Music lovers who enjoy songs in languages they don't fully understand (Urdu, Hindi, Punjabi, Arabic, Spanish, etc.) miss the deeper meaning and emotional resonance of the lyrics. Simple translation tools give literal word-for-word output that loses the poetry.

## Target user

- Music enthusiasts who listen to international music
- Second-generation immigrants who partially understand their parents' native language
- Anyone curious about the meaning behind a song they love

## Core user flow

```
User uploads/records audio
        ↓
System transcribes audio to text (in original language)
        ↓
System translates to English (preserving poetic structure)
        ↓
System interprets deeper meaning (metaphors, cultural context, emotion)
        ↓
User sees: Original lyrics | English translation | Deep interpretation
```

## Key features (v1)

### Must have
- [ ] Audio input (upload file or record)
- [ ] Transcription of vocals to text
- [ ] Language detection
- [ ] Translation to English
- [ ] Meaning interpretation with cultural context

### Nice to have
- [ ] Line-by-line breakdown
- [ ] Highlight metaphors and literary devices
- [ ] Link to cultural/historical references
- [ ] Save and share interpretations

## Technical considerations

### Transcription
- Need a model that handles non-English audio well
- Whisper is strong for multilingual transcription
- May need to handle background music (vocal isolation?)

### Translation
- Standard translation APIs lose poetic nuance
- Better approach: Use LLM with prompt like "Translate this song preserving poetic structure and emotional tone"

### Interpretation
- This is the core value-add
- Need prompts that encourage cultural context, metaphor identification, and emotional analysis
- May need language-specific context (Urdu poetry has specific conventions like "ghazal")

## Success metrics

- User completes full flow (uploads → sees interpretation)
- User rates interpretation as "helpful" or "accurate"
- User shares or saves result

## Open questions

1. How to handle songs with mixed languages (code-switching)?
2. Should we build vocal isolation or require clean audio?
3. How do we validate interpretation accuracy for languages we don't speak?
