import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from config.settings import settings

@dataclass
class VisualElement:
    """Represents a visual element extracted from narration"""
    type: str  # 'character', 'setting', 'object', 'action', 'mood'
    description: str
    importance: float  # 0.0 to 1.0
    context: str  # surrounding text for context

@dataclass
class SceneDescription:
    """Complete scene description for generation"""
    main_prompt: str
    visual_elements: List[VisualElement]
    scene_type: str  # 'static', 'dynamic', 'transition'
    mood: str
    style_hints: List[str]
    negative_prompt: str
    generation_type: str  # 'image' or 'video'

class TextExtractor:
    """Extracts visual elements and scene descriptions from narration text"""
    
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        
        # Keywords for different visual element types
        self.character_keywords = [
            'character', 'person', 'man', 'woman', 'child', 'hero', 'villain',
            'knight', 'wizard', 'princess', 'king', 'queen', 'warrior', 'maiden'
        ]
        
        self.setting_keywords = [
            'castle', 'forest', 'mountain', 'village', 'city', 'ocean', 'desert',
            'cave', 'palace', 'tower', 'garden', 'bridge', 'valley', 'meadow',
            'dungeon', 'library', 'chamber', 'hall', 'courtyard', 'battlefield'
        ]
        
        self.object_keywords = [
            'sword', 'shield', 'crown', 'ring', 'book', 'scroll', 'crystal',
            'staff', 'wand', 'armor', 'cloak', 'pendant', 'gem', 'treasure',
            'door', 'window', 'throne', 'table', 'candle', 'fire'
        ]
        
        self.action_keywords = [
            'walking', 'running', 'flying', 'fighting', 'dancing', 'singing',
            'casting', 'summoning', 'riding', 'climbing', 'falling', 'jumping',
            'opening', 'closing', 'entering', 'leaving', 'approaching', 'hiding'
        ]
        
        self.mood_keywords = {
            'dark': ['dark', 'gloomy', 'ominous', 'sinister', 'evil', 'scary', 'haunting'],
            'bright': ['bright', 'cheerful', 'happy', 'joyful', 'radiant', 'glowing'],
            'mysterious': ['mysterious', 'enigmatic', 'secretive', 'hidden', 'ancient'],
            'dramatic': ['dramatic', 'intense', 'powerful', 'epic', 'grand', 'majestic'],
            'peaceful': ['peaceful', 'calm', 'serene', 'tranquil', 'gentle', 'quiet'],
            'romantic': ['romantic', 'beautiful', 'elegant', 'graceful', 'lovely']
        }
        
        # Style mappings for different genres/themes
        self.style_mappings = {
            'fantasy': 'fantasy art, magical, ethereal, mystical',
            'medieval': 'medieval, historical, classical painting style',
            'modern': 'modern, contemporary, realistic',
            'anime': 'anime style, manga, Japanese animation',
            'realistic': 'photorealistic, detailed, high resolution',
            'artistic': 'artistic, painterly, impressionistic'
        }
        
        # Initialize LLM if requested (optional for better extraction)
        self.llm_pipeline = None
        if use_llm and torch.cuda.is_available():
            try:
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",  # Lightweight model
                    device=0 if settings.device == "cuda" else -1,
                    torch_dtype=torch.float16 if settings.device == "cuda" else torch.float32
                )
            except Exception as e:
                print(f"Could not load LLM for text extraction: {e}")
                self.use_llm = False
    
    def extract_scene_description(self, text: str, context_history: List[str] = None) -> SceneDescription:
        """Extract complete scene description from narration text"""
        
        # Clean and prepare text
        clean_text = self._clean_text(text)
        
        # Simple approach: detect key elements directly from text
        visual_elements = self._detect_key_elements(clean_text)
        
        # Determine scene type and mood
        scene_type = self._determine_scene_type(clean_text)
        mood = self._extract_mood(clean_text)
        
        # Generate main prompt from the actual story content
        main_prompt = self._generate_natural_prompt(clean_text, mood)
        
        # Determine generation type
        generation_type = self._determine_generation_type(clean_text)
        
        # Generate style hints
        style_hints = self._generate_style_hints(clean_text, mood)
        
        # Generate negative prompt
        negative_prompt = self._generate_negative_prompt(mood)
        
        return SceneDescription(
            main_prompt=main_prompt,
            visual_elements=visual_elements,
            scene_type=scene_type,
            mood=mood,
            style_hints=style_hints,
            negative_prompt=negative_prompt,
            generation_type=generation_type
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common speech artifacts
        text = re.sub(r'\b(um|uh|er|ah)\b', '', text, flags=re.IGNORECASE)
        
        # Remove repeated words
        words = text.split()
        cleaned_words = []
        prev_word = None
        
        for word in words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
            prev_word = word.lower()
        
        return ' '.join(cleaned_words)
    
    def _detect_key_elements(self, text: str) -> List[VisualElement]:
        """Detect key visual elements naturally from the text"""
        elements = []
        words = text.lower().split()
        
        # Look for nouns that are likely visual subjects
        import re
        
        # Find potential characters/creatures (nouns that could be subjects)
        character_patterns = [
            r'\b(dragon|knight|wizard|king|queen|princess|prince|warrior|beast|creature|monster|giant|fairy|elf|dwarf|witch|sorcerer|mage|demon|angel|spirit|person|man|woman|child|hero|villain)\b',
            r'\b(goblin|orc|troll|vampire|werewolf|phoenix|unicorn|centaur|griffin|pegasus)\b'
        ]
        
        for pattern in character_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                elements.append(VisualElement(
                    type='character',
                    description=match,
                    importance=0.9,
                    context=self._extract_context(text, match)
                ))
        
        # Find settings/locations
        setting_patterns = [
            r'\b(mountain|mountains|castle|forest|palace|tower|village|city|ocean|sea|desert|cave|valley|meadow|bridge|garden|dungeon|library|chamber|hall|courtyard|battlefield|kingdom|realm)\b',
            r'\b(sky|clouds|stars|moon|sun|field|hill|hills|cliff|river|lake|pond|island)\b'
        ]
        
        for pattern in setting_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                elements.append(VisualElement(
                    type='setting',
                    description=match,
                    importance=0.8,
                    context=self._extract_context(text, match)
                ))
        
        # Find notable objects or features
        object_patterns = [
            r'\b(fire|flame|sword|shield|crown|ring|armor|cloak|staff|wand|crystal|gem|treasure|door|window|throne)\b',
            r'\b(wings|scales|claws|teeth|eyes|breath|magic|spell|power)\b'
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                elements.append(VisualElement(
                    type='object',
                    description=match,
                    importance=0.6,
                    context=self._extract_context(text, match)
                ))
        
        return elements
    
    def _extract_context(self, text: str, keyword: str, window: int = 20) -> str:
        """Extract context around a keyword"""
        words = text.split()
        keyword_lower = keyword.lower()
        
        for i, word in enumerate(words):
            if keyword_lower in word.lower():
                start = max(0, i - window // 2)
                end = min(len(words), i + window // 2 + 1)
                return ' '.join(words[start:end])
        
        return text[:100]  # Fallback to first 100 chars
    
    def _determine_scene_type(self, text: str) -> str:
        """Determine if scene should be static image or dynamic video"""
        action_words = ['moving', 'walking', 'running', 'flying', 'dancing', 'fighting', 'flowing', 'riding', 'climbing', 'jumping', 'breathing', 'spitting', 'attacking', 'swooping']
        transition_words = ['then', 'next', 'suddenly', 'meanwhile', 'after', 'before', 'while', 'as']
        
        text_lower = text.lower()
        action_count = sum(1 for word in action_words if word in text_lower)
        transition_count = sum(1 for word in transition_words if text_lower.count(word) >= 1)
        
        # Check for motion verbs
        motion_verbs = ['flies', 'runs', 'walks', 'moves', 'dances', 'fights', 'rides', 'climbs', 'jumps', 'breathes', 'spits']
        motion_count = sum(1 for verb in motion_verbs if verb in text_lower)
        
        if action_count >= 2 or motion_count >= 1:
            return 'dynamic'
        elif transition_count >= 1:
            return 'transition'
        else:
            return 'static'
    
    def _extract_mood(self, text: str) -> str:
        """Extract mood from text"""
        mood_scores = {}
        
        for mood, keywords in self.mood_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text.lower())
            if score > 0:
                mood_scores[mood] = score
        
        if mood_scores:
            return max(mood_scores, key=mood_scores.get)
        else:
            return 'neutral'
    
    def _generate_natural_prompt(self, text: str, mood: str) -> str:
        """Generate natural prompt directly from story content - completely generalistic"""
        
        # Clean the text
        clean_text = text.lower()
        
        # Remove only the most basic filler words that add no visual value
        filler_words = ['there once was', 'once upon a time', 'and then', 'so that', 'would', 'could', 'should', 'that', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by']
        for filler in filler_words:
            clean_text = clean_text.replace(' ' + filler + ' ', ' ')
        
        # Remove punctuation and get all meaningful words
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_text)  # Words 3+ chars
        
        # Filter out common non-visual words (keep it concise for production)
        skip_words = {
            # Articles, prepositions, conjunctions
            'the', 'a', 'an', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            # Common verbs that don't add visual value
            'was', 'were', 'had', 'has', 'have', 'been', 'be', 'is', 'are', 'am', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'do', 'does', 'did', 'get', 'got', 'take', 'took', 'give', 'gave', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew', 'think', 'thought', 'say', 'said', 'tell', 'told', 'make', 'made', 'let', 'put',
            # Pronouns and determiners  
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those', 'some', 'any', 'all', 'each', 'every', 'no', 'none', 'one', 'two', 'first', 'other', 'another', 'such', 'only', 'own', 'same',
            # Common adverbs and modifiers
            'very', 'really', 'quite', 'just', 'still', 'also', 'too', 'so', 'more', 'most', 'much', 'many', 'few', 'little', 'less', 'least', 'well', 'better', 'best', 'worse', 'worst', 'good', 'bad', 'new', 'old', 'long', 'short', 'high', 'low', 'right', 'left', 'next', 'last', 'first', 'second', 'both', 'either', 'neither',
            # Question words and relatives
            'what', 'when', 'where', 'who', 'whom', 'whose', 'which', 'why', 'how', 'that', 'who', 'what', 'where', 'when', 'why', 'how',
            # Time and frequency
            'now', 'then', 'here', 'there', 'today', 'yesterday', 'tomorrow', 'never', 'always', 'sometimes', 'often', 'usually', 'soon', 'already', 'yet', 'still', 'again', 'once', 'twice',
            # Discourse markers
            'yes', 'no', 'not', 'perhaps', 'maybe', 'probably', 'certainly', 'sure', 'indeed', 'however', 'therefore', 'because', 'since', 'although', 'though', 'unless', 'if', 'whether', 'while', 'until', 'as'
        }
        
        # Keep only visual/descriptive words
        visual_words = []
        for word in words:
            if word not in skip_words and len(word) >= 3:
                visual_words.append(word)
        
        # Take the most important words (limit to prevent overwhelming the prompt)
        important_words = visual_words[:10]
        
        # Add mood-based atmosphere
        if mood and mood != 'neutral':
            important_words.append(f"{mood} atmosphere")
        
        # Add basic quality terms
        important_words.extend(["detailed", "high quality", "cinematic lighting"])
        
        return ', '.join(important_words)
    
    def _determine_generation_type(self, text: str) -> str:
        """Determine whether to generate image or video based on content"""
        # Check for strong action/motion indicators
        action_indicators = ['spit', 'spits', 'breathing', 'flying', 'moving', 'running', 'attacking', 'fighting']
        
        text_lower = text.lower()
        has_strong_action = any(action in text_lower for action in action_indicators)
        
        # For now, prefer images for better quality and speed
        # Can be overridden by user in the interface
        if has_strong_action:
            return 'video'
        else:
            return 'image'
    
    def _generate_style_hints(self, text: str, mood: str) -> List[str]:
        """Generate style hints based on content"""
        hints = []
        
        # Genre detection
        if any(word in text.lower() for word in ['magic', 'wizard', 'dragon', 'castle']):
            hints.append(self.style_mappings['fantasy'])
        elif any(word in text.lower() for word in ['knight', 'medieval', 'sword']):
            hints.append(self.style_mappings['medieval'])
        
        # Mood-based style
        if mood == 'dark':
            hints.append("dark fantasy, gothic")
        elif mood == 'bright':
            hints.append("vibrant colors, bright lighting")
        elif mood == 'mysterious':
            hints.append("mysterious lighting, shadows")
        
        return hints
    
    def _generate_negative_prompt(self, mood: str) -> str:
        """Generate negative prompt to avoid unwanted elements"""
        negative_elements = [
            "blurry", "low quality", "distorted", "ugly", "bad anatomy",
            "poorly drawn", "extra limbs", "duplicate", "mutation", "deformed"
        ]
        
        if mood == 'peaceful':
            negative_elements.extend(["violence", "weapons", "blood", "war"])
        elif mood == 'bright':
            negative_elements.extend(["dark", "gloomy", "horror"])
        elif mood == 'dark':
            negative_elements.extend(["bright", "cheerful", "happy"])
        
        return ', '.join(negative_elements)

# Example usage
if __name__ == "__main__":
    extractor = TextExtractor()
    
    test_text = "The brave knight walked through the dark forest towards the ancient castle, his sword gleaming in the moonlight."
    
    scene = extractor.extract_scene_description(test_text)
    
    print(f"Main Prompt: {scene.main_prompt}")
    print(f"Scene Type: {scene.scene_type}")
    print(f"Generation Type: {scene.generation_type}")
    print(f"Mood: {scene.mood}")
    print(f"Visual Elements: {len(scene.visual_elements)}")
    for element in scene.visual_elements:
        print(f"  - {element.type}: {element.description} (importance: {element.importance})")