"""
AI Service for image analysis using LM Studio
Handles communication with local LM Studio API
"""

import requests
import base64
import json
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


class AIService:
    # Common JSON format instructions used across all prompts
    CRITICAL_JSON_INSTRUCTIONS = """
You must respond with ONLY a valid JSON object in this exact format:
{
  "description": "your description here",
  "tags": ["tag1", "tag2", "tag3"],
  "suggested_filename": "descriptive_filename_here"
}

CRITICAL INSTRUCTIONS:
- Your ENTIRE response must be ONLY the JSON object above
- Do NOT add any explanations before or after the JSON
- Do NOT use markdown code blocks (no ```json```)
- Do NOT add any commentary or additional text
- Just the raw JSON object and nothing else
- Ensure the JSON is valid with no trailing commas or syntax errors"""
    def __init__(self, lm_studio_url: str = "http://localhost:1234"):
        self.lm_studio_url = lm_studio_url
        self.api_endpoint = f"{lm_studio_url}/v1/chat/completions"

        # Different description styles/prompts
        self.prompts = {
            'classic': {
                'name': 'Classic',
                'description': 'Detailed and comprehensive (2-4 sentences)',
                'prompt': """Analyze this image thoroughly and provide:
1. A detailed description (2-4 sentences) that captures:
   - Main subject(s) and what they're doing
   - Art style/medium (e.g., photograph, comic art, illustration, cartoon, painting, 3D render, etc.)
   - Notable visual features (clothing, accessories, tattoos, piercings, hair style/color, etc.)
   - Setting/environment and atmosphere
   - Lighting and color palette if notable
2. 8-15 comprehensive tags covering:
   - Subject type (woman, man, couple, animal, landscape, etc.)
   - Art style (comic, illustration, photo, cartoon, anime, etc.)
   - Specific features (tattoos, sword, urban, nature, etc.)
   - Mood/atmosphere (dramatic, peaceful, dark, vibrant, etc.)
   - Colors if prominent (monochrome, colorful, blue_tones, etc.)
3. A suggested filename (descriptive, lowercase, use underscores, no spaces, max 50 chars, WITHOUT file extension)

Guidelines for description:
- STRICT LIMIT: 2-4 sentences MAXIMUM - be concise!
- Be specific about art style (very important!) - is it a photo, comic art, illustration, digital art, painting?
- ONLY describe what IS present - do NOT mention what is NOT there (no "without makeup", "no accessories", etc.)
- Include notable details like tattoos, piercings, unique clothing, weapons, etc.
- Describe the setting and mood
- Focus on POSITIVE attributes - what makes the image interesting
- Keep it factual and comprehensive but concise (2-4 sentences max)

Guidelines for tags:
- ALWAYS include art style tags (comic, illustration, photo, cartoon, anime, painted, etc.)
- Include subject descriptors (woman, man, portrait, landscape, etc.)
- Include specific features (tattoos, sword, leather, cyberpunk, etc.)
- Include mood/atmosphere tags (dark, dramatic, peaceful, etc.)
- NEVER use negative tags (no "no_mood", "no_accessories", "without_x", etc.)
- Tags must describe WHAT IS PRESENT, not what is absent
- Keep tags lowercase, prefer single words or compound_words
- Ensure they are unique and highly relevant
- Aim for 8-15 tags for comprehensive categorization
- Every tag must add VALUE - avoid generic filler tags

Guidelines for filename:
- Should reflect main subject and key characteristics
- Use underscores instead of spaces
- Keep it descriptive but under 50 chars
- Do NOT include file extension
- Example: "woman_with_sword_comic_style" not just "woman_with_sword\""""
            },

            'artistic': {
                'name': 'Artistic',
                'description': 'Detailed, poetic, and creative description',
                'prompt': """Analyze this image with artistic detail and provide:
1. A detailed, artistic description (3-4 sentences) that captures:
   - Art style/medium (photograph, comic art, illustration, digital art, painting, etc.) - CRITICAL!
   - Main subjects and their characteristics (features, clothing, accessories, tattoos, etc.)
   - Mood, atmosphere, colors, composition, and emotional impact
   - Lighting, shadows, and visual techniques
   - Be poetic and evocative while being accurate
2. 10-15 comprehensive tags including:
   - Art style (comic, illustration, photo, painted, etc.) - REQUIRED
   - Subject type (woman, man, portrait, etc.)
   - Specific features (tattoos, piercings, weapons, etc.)
   - Mood and atmosphere
   - Technical and artistic aspects
3. A suggested filename (descriptive, lowercase, use underscores, no spaces, max 50 chars, WITHOUT file extension)

Guidelines:
- LIMIT: 3-4 sentences for description - be evocative but concise
- ALWAYS identify and mention the art style/medium in description
- ONLY describe what IS visible - do NOT mention absences (no "lacks", "without", "no", etc.)
- Make the description rich, detailed, and atmospheric
- Include details about lighting, composition, mood, colors
- Use vivid, descriptive language for PRESENT elements
- Tags MUST include art style (comic, illustration, photo, cartoon, anime, painted, etc.)
- NEVER use negative tags (no "no_mood", "without_x", "lacks_y", etc.)
- Tags should include artistic and technical terms; keep them lowercase, prefer single words or compound_words
- Every tag must describe something PRESENT in the image
- Ensure they are unique and relevant
- Filename should be descriptive but concise
- Use underscores instead of spaces in filename
- Do NOT include file extension in suggested_filename"""
            },

            'spicy': {
                'name': 'Spicy',
                'description': 'Provocative and attention-grabbing style',
                'prompt': """Analyze this image with a bold, provocative style and provide:
1. A captivating description (3-4 sentences) that includes:
   - Art style/medium (photo, comic, illustration, etc.) - REQUIRED
   - Main subjects and striking visual features (tattoos, accessories, unique style, etc.)
   - Attention-grabbing, bold, and engaging language
   - Emphasis on the most striking and alluring aspects
2. 10-15 tags including:
   - Art style (comic, illustration, photo, etc.) - REQUIRED
   - Subject type and specific features
   - Mood and aesthetic qualities
   - Visual characteristics
3. A suggested filename (descriptive, evocative, lowercase, use underscores, no spaces, max 50 chars, WITHOUT file extension)

Guidelines:
- LIMIT: 3-4 sentences - be bold but concise
- ALWAYS identify art style/medium first (comic, photo, illustration, etc.)
- ONLY describe visible elements - do NOT mention what's missing
- Make the description bold, captivating, and more provocative with sensual, tantalizing language to heighten allure and intensity
- Emphasize visual appeal and striking elements that ARE PRESENT
- Use confident, engaging language
- Focus on what makes the image stand out
- Tags MUST include art style (comic, illustration, photo, cartoon, anime, etc.)
- NEVER use negative tags (no "no_x", "without_y", "lacks_z", etc.)
- Tags should include mood and aesthetics; keep them lowercase, prefer single words or compound_words
- Every tag must add value and describe something VISIBLE
- Ensure they are unique and relevant
- Filename should be descriptive but concise
- Use underscores instead of spaces in filename
- Do NOT include file extension in suggested_filename"""
            },

            'social': {
                'name': 'Social Media',
                'description': 'Optimized for Instagram, Facebook, Twitter',
                'prompt': """Analyze this image for social media posting and provide:
1. A social media-ready description (2-4 paragraphs) that includes:
   - Art style/type (photo, comic, illustration, digital art, etc.) - mention naturally
   - Main subject and notable features (tattoos, style, unique elements, etc.)
   - Engaging, relatable tone perfect for Instagram, Facebook, or Twitter
   - Conversational and authentic language
2. 10-15 trending hashtags and relevant keywords including:
   - Art style hashtags (#ComicArt, #Illustration, #Photography, etc.)
   - Subject hashtags (#Portrait, #Fashion, etc.)
   - Feature hashtags (#Tattoos, #InkedGirls, etc.)
   - Trending and relevant keywords
3. A suggested filename (catchy, descriptive, lowercase, use underscores, no spaces, max 50 chars, WITHOUT file extension)

Guidelines:
- Naturally mention art style in description (e.g., "This stunning comic art shows...")
- Write in a friendly, conversational tone
- Make it shareable and relatable
- Tags MUST include art style hashtags (#ComicArt, #Illustration, #DigitalArt, etc.)
- Tags should include the # for hashtags where appropriate; keep them lowercase, prefer single words or phrases
- Ensure they are unique and relevant
- Consider what would perform well on social platforms
- Filename should be descriptive but concise
- Use underscores instead of spaces in filename
- Do NOT include file extension in suggested_filename"""
            },

            'tags': {
                'name': 'Tags Only',
                'description': 'Generate only tags/keywords without description',
                'prompt': """Analyze this image and provide ONLY comprehensive tags/keywords.

SPECIAL INSTRUCTIONS:
- Generate 10-15 relevant, descriptive tags/keywords for this image
- Leave description empty (empty string "")
- Leave suggested_filename empty (empty string "")

Guidelines for tags - MUST include:
1. Art style (comic, illustration, photo, cartoon, anime, painted, digital_art, etc.) - REQUIRED as first tags
2. Subject type (woman, man, portrait, landscape, animal, etc.)
3. Specific features (tattoos, piercings, sword, accessories, etc.)
4. Mood/atmosphere (dark, dramatic, peaceful, vibrant, etc.)
5. Colors if prominent (monochrome, colorful, blue_tones, etc.)
6. Setting (urban, nature, indoor, studio, etc.)

Format rules:
- Keep tags lowercase
- Prefer single words (use compound_words if needed like "comic_art" or "golden_hour")
- ALWAYS include art style tags (comic, illustration, photo, etc.) - this is CRITICAL
- Include specific details like tattoos, weapons, clothing style, etc.
- NEVER use negative tags (no "no_mood", "no_accessories", "without_x", etc.)
- Tags must describe WHAT IS PRESENT, not what is absent
- Every tag must add VALUE - no filler tags
- Ensure tags are unique and highly relevant
- No hashtags (#) - just plain keywords
- Aim for 10-15 comprehensive tags"""
            },

            'custom': {
                'name': 'Custom',
                'description': 'Use your own custom prompt',
                'prompt': None  # Will be provided by user
            }
        }

    def get_available_styles(self) -> Dict[str, Dict]:
        """Get all available description styles"""
        return {
            key: {
                'name': value['name'],
                'description': value['description']
            }
            for key, value in self.prompts.items()
        }

    def check_connection(self) -> Tuple[bool, str]:
        """Check if LM Studio is running and accessible"""
        try:
            response = requests.get(f"{self.lm_studio_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return True, "LM Studio is connected"
            else:
                return False, f"LM Studio returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to LM Studio. Is it running?"
        except requests.exceptions.Timeout:
            return False, "Connection to LM Studio timed out"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def analyze_image(self, image_path: str, style: str = 'classic', custom_prompt: str = None) -> Optional[Dict]:
        """
        Analyze image and return description and tags

        Args:
            image_path: Path to the image file
            style: Description style ('classic', 'artistic', 'spicy', 'social', 'custom')
            custom_prompt: Custom prompt text (only used if style='custom')

        Returns: {'description': str, 'tags': List[str], 'suggested_filename': str} or None on error
        """
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            base64_image = base64.b64encode(image_data).decode('utf-8')

            # Determine image format
            ext = Path(image_path).suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp'
            }.get(ext, 'image/jpeg')

            # Get prompt based on style
            if style == 'custom' and custom_prompt:
                # Wrap custom prompt with JSON instructions
                prompt = f"{custom_prompt}\n\n{self.CRITICAL_JSON_INSTRUCTIONS}"
            elif style in self.prompts:
                prompt = f"{self.prompts[style]['prompt']}\n\n{self.CRITICAL_JSON_INSTRUCTIONS}"
            else:
                # Fallback to classic
                prompt = f"{self.prompts['classic']['prompt']}\n\n{self.CRITICAL_JSON_INSTRUCTIONS}"

            logger.info("Using '%s' style for analysis", style)

            # Prepare API request
            payload = {
                "model": "llava",  # or whatever vision model is loaded
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            logger.debug("Sending analysis request to %s", self.api_endpoint)

            # Send request to LM Studio
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=120  # 2 minutes timeout for slow models
            )

            logger.debug("Response status: %d", response.status_code)

            if response.status_code == 200:
                result = response.json()

                # Extract content from response
                if 'choices' not in result or len(result['choices']) == 0:
                    logger.error("Invalid response structure from LM Studio")
                    return None

                content = result['choices'][0]['message']['content']
                logger.debug("AI response (truncated): %s...", content[:200])

                # Try to parse JSON from content
                parsed = self._extract_json(content)

                if parsed:
                    result = {
                        'description': parsed.get('description', ''),
                        'tags': parsed.get('tags', []),
                        'suggested_filename': parsed.get('suggested_filename', '')
                    }
                    logger.info("AI suggested filename: %s", result.get('suggested_filename', 'none'))
                    return result
                else:
                    # Fallback: treat whole response as description
                    logger.warning("Could not parse JSON from AI response, using raw response as description")
                    return {
                        'description': content.strip(),
                        'tags': [],
                        'suggested_filename': '',
                        'warning': 'AI did not return valid JSON. The raw response is in "description".'
                    }
            else:
                logger.error("LM Studio error: %d - %s", response.status_code, response.text)
                return None

        except FileNotFoundError:
            logger.error("Image file not found: %s", image_path)
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error("Connection error: %s. Make sure LM Studio is running with local server enabled", str(e))
            return None
        except requests.exceptions.Timeout:
            logger.error("Analysis timed out for %s", image_path)
            return None
        except Exception as e:
            logger.error("Error analyzing image: %s", str(e), exc_info=True)
            return None
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from text that might contain markdown code blocks or extra text.
        Handles cases where AI returns JSON followed by additional explanation.
        """
        import re

        # Try direct parse first (if text is pure JSON)
        text = text.strip()
        try:
            return json.loads(text)
        except:
            pass

        # Simple approach: find first { and last } - covers most common cases
        # This is fast and handles cases where AI adds comments after JSON
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]

            # Aggressive cleanup: strip markdown code blocks even within extracted JSON
            # This handles cases where model adds ```json``` between { and }
            json_str = json_str.strip()
            json_str = json_str.lstrip('`').rstrip('`')  # Remove backticks
            if json_str.startswith('json'):
                json_str = json_str[4:].lstrip()  # Remove 'json' marker
            json_str = json_str.lstrip('`').rstrip('`')  # Remove remaining backticks

            try:
                parsed = json.loads(json_str)
                logger.debug("Successfully extracted JSON using find/rfind method")
                return parsed
            except:
                pass

        # Look for ```json ... ``` markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # Look for properly nested JSON {...} block
        # This regex handles nested objects correctly
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass

        # Last resort: Manual brace counting to extract complete JSON
        # This handles deeply nested structures with proper escaping
        try:
            start_idx = text.find('{')
            if start_idx == -1:
                return None

            brace_count = 0
            end_idx = start_idx
            in_string = False
            escape_next = False

            for i in range(start_idx, len(text)):
                char = text[i]

                # Handle string escaping
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                # Track if we're inside a string
                if char == '"':
                    in_string = not in_string
                    continue

                # Only count braces outside of strings
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

            if end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                parsed = json.loads(json_str)
                logger.debug("Successfully extracted JSON using brace counting")
                return parsed
        except Exception as e:
            logger.debug("Brace counting extraction failed: %s", str(e))
            pass

        logger.warning("Could not extract valid JSON from AI response")
        return None
    
    def batch_analyze(self, image_paths: list, progress_callback=None) -> Dict[str, Dict]:
        """
        Analyze multiple images
        Returns: {image_path: {'description': str, 'tags': list}, ...}
        """
        results = {}
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, total, path)

            result = self.analyze_image(path)
            results[path] = result

        return results

    def suggest_boards(self, image_description: str, image_tags: list,
                       existing_boards: list) -> Optional[Dict]:
        """
        Suggest which board(s) an image should belong to based on its description and tags.
        Can also suggest creating a new board.

        Args:
            image_description: The AI-generated description of the image
            image_tags: List of tags for the image
            existing_boards: List of dicts with 'id', 'name', and 'description' of existing boards

        Returns:
            {
                'action': 'add_to_existing' or 'create_new',
                'suggested_boards': [board_id1, board_id2, ...],  # if action is 'add_to_existing'
                'confidence': 0.0-1.0,  # how confident AI is about the suggestion
                'new_board': {  # if action is 'create_new'
                    'name': 'suggested name',
                    'description': 'suggested description'
                },
                'reasoning': 'explanation of why these boards were suggested'
            }
        """
        try:
            # Prepare simplified board list (flat, no verbose hierarchy)
            def format_board_simple(board, indent=0):
                # Simple format: ID: N, Name: 'X', Desc: 'Y' [Parent: Z]
                prefix = "  " * indent
                board_text = f"{prefix}ID: {board['id']}, Name: '{board['name']}'"
                if board.get('description'):
                    board_text += f", Desc: '{board['description'][:50]}...'" if len(board.get('description', '')) > 50 else f", Desc: '{board['description']}'"
                if board.get('parent_id'):
                    board_text += f" [Parent: {board['parent_id']}]"

                result = [board_text]

                # Add sub-boards recursively
                if board.get('sub_boards'):
                    for sub_board in board['sub_boards']:
                        result.extend(format_board_simple(sub_board, indent + 1))

                return result

            boards_info = []
            for board in existing_boards:
                if not board.get('parent_id'):  # Only process top-level boards
                    boards_info.extend(format_board_simple(board))

            boards_context = "\n".join(boards_info) if boards_info else "No boards"
            tags_text = ", ".join(image_tags) if image_tags else "No tags"

            # Simplified, direct prompt - minimal noise
            prompt = f"""TASK: Assign image to existing board ID or propose new sub-board/top-level board.

IMAGE DATA:
Description: {image_description}
Tags: {tags_text}

EXISTING BOARDS (ID: Name):
{boards_context}

RULES:
1. PREFER creating SUB-BOARD if GENERAL category exists (e.g., Image=Tattoo, Board=Woman → New SUB-BOARD 'Tattoos' under Woman's ID).
2. Use "create_new" for sub-board or new top-level board.
3. Use "add_to_existing" ONLY if perfectly matches existing board.
4. Confidence must be 0.0-1.0.

REQUIRED OUTPUT FORMAT (RAW JSON ONLY):
{{
  "action": "add_to_existing" or "create_new",
  "suggested_boards": [ID_1, ID_2],
  "confidence": 0.0-1.0,
  "new_board": {{
    "name": "Sub-Category Name",
    "description": "Brief description",
    "parent_id": 123 or null
  }} or null,
  "reasoning": "1-2 sentence explanation."
}}

CRITICAL:
- Response MUST be RAW JSON ONLY.
- NO explanations, NO markdown blocks (no ```json```).
- Just the JSON object."""

            # Send request to LM Studio (text-only, no image needed)
            payload = {
                "model": "llava",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.5  # Lower temperature for more consistent suggestions
            }

            logger.debug("Requesting board suggestions from LM Studio...")

            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                if 'choices' not in result or len(result['choices']) == 0:
                    logger.error("Invalid response structure from LM Studio for board suggestion")
                    return None

                content = result['choices'][0]['message']['content']
                logger.debug("AI board suggestion response (truncated): %s...", content[:200])

                # Parse JSON response
                parsed = self._extract_json(content)

                if parsed:
                    # Validate and clean up response
                    suggestion = {
                        'action': parsed.get('action', 'create_new'),
                        'suggested_boards': parsed.get('suggested_boards', []),
                        'confidence': float(parsed.get('confidence', 0.5)),
                        'new_board': parsed.get('new_board'),
                        'reasoning': parsed.get('reasoning', '')
                    }

                    logger.info("Board suggestion: %s (confidence: %.2f)", suggestion['action'], suggestion['confidence'])
                    return suggestion
                else:
                    logger.warning("Could not parse JSON response for board suggestion")
                    return None
            else:
                logger.error("LM Studio error for board suggestion: %d - %s", response.status_code, response.text)
                return None

        except Exception as e:
            logger.error("Error suggesting boards: %s", str(e), exc_info=True)
            return None