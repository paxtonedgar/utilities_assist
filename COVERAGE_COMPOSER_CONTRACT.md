# Composer Contract Implementation

## Citation Enforcement Requirement

According to specification #4, the composer (response generator) must enforce citation format:

```
- Do X in Jira: Project KEY → Request Type "Y". [Doc §Onboarding → Request Access](URL#Onboarding)

If any step lacks a citation, drop it.
```

## Implementation Location

**File**: `src/services/respond.py`

**Function**: The response generation functions that build final answers from selected passages.

## Required Changes

1. **Passage Metadata**: Ensure selected passages include:
   - `url`: Source URL
   - `heading`: Section heading for anchor links
   - `title`: Document title

2. **Citation Format**: Each bullet point must include:
   - Action description
   - Citation with format `[Doc §Heading → Subheading](URL#anchor)`
   - Deep links using `url + "#" + anchor_sanitized`

3. **Citation Validation**: 
   - If a step lacks proper citation, drop the entire step
   - Log dropped steps for debugging

## Integration with Coverage Gate

The coverage gate provides `selected_passages` with proper metadata:
- `passage["heading"]` → used for anchor links
- `passage["url"]` → used for base URL  
- `passage["title"]` → used for document reference

## Implementation Status

⚠️ **NOT YET IMPLEMENTED** - This is a manual integration step that requires:
1. Modifying the response generation templates
2. Adding citation validation logic
3. Testing with actual passage metadata

This completes the specification but requires coordination with the existing response generation system.