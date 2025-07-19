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
        
        # Extract visual elements
        visual_elements = self._extract_visual_elements(clean_text)
        
        # Determine scene type
        scene_type = self._determine_scene_type(clean_text, visual_elements)
        
        # Extract mood
        mood = self._extract_mood(clean_text)
        
        # Generate main prompt
        main_prompt = self._generate_main_prompt(clean_text, visual_elements, mood)
        
        # Determine generation type
        generation_type = self._determine_generation_type(scene_type, visual_elements)
        
        # Generate style hints
        style_hints = self._generate_style_hints(clean_text, mood)
        
        # Generate negative prompt
        negative_prompt = self._generate_negative_prompt(scene_type, mood)
        
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
    
    def _extract_visual_elements(self, text: str) -> List[VisualElement]:
        """Extract visual elements from text using keyword matching"""
        elements = []
        words = text.lower().split()
        
        # Extract characters
        for keyword in self.character_keywords:
            if keyword in text.lower():
                context = self._extract_context(text, keyword)
                elements.append(VisualElement(
                    type='character',
                    description=keyword,
                    importance=0.8,
                    context=context
                ))
        
        # Extract settings
        for keyword in self.setting_keywords:
            if keyword in text.lower():
                context = self._extract_context(text, keyword)
                elements.append(VisualElement(
                    type='setting',
                    description=keyword,
                    importance=0.9,
                    context=context
                ))
        
        # Extract objects
        for keyword in self.object_keywords:
            if keyword in text.lower():
                context = self._extract_context(text, keyword)
                elements.append(VisualElement(
                    type='object',
                    description=keyword,
                    importance=0.6,
                    context=context
                ))
        
        # Extract actions
        for keyword in self.action_keywords:
            if keyword in text.lower():
                context = self._extract_context(text, keyword)
                elements.append(VisualElement(
                    type='action',
                    description=keyword,
                    importance=0.7,
                    context=context
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
    
    def _determine_scene_type(self, text: str, elements: List[VisualElement]) -> str:
        """Determine if scene should be static image or dynamic video"""
        action_words = ['moving', 'walking', 'running', 'flying', 'dancing', 'fighting', 'flowing']
        transition_words = ['then', 'next', 'suddenly', 'meanwhile', 'after', 'before']
        
        action_count = sum(1 for word in action_words if word in text.lower())
        transition_count = sum(1 for word in transition_words if word in text.lower())
        
        # Count action elements
        action_elements = len([e for e in elements if e.type == 'action'])
        
        if action_count >= 2 or action_elements >= 2:
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
    
    def _generate_main_prompt(self, text: str, elements: List[VisualElement], mood: str) -> str:
        """Generate main prompt for image/video generation"""
        
        # Start with key visual elements
        key_elements = sorted(elements, key=lambda x: x.importance, reverse=True)[:3]
        
        prompt_parts = []
        
        # Add setting first (most important for composition)
        setting_elements = [e for e in key_elements if e.type == 'setting']
        if setting_elements:
            prompt_parts.append(setting_elements[0].description)
        
        # Add characters
        character_elements = [e for e in key_elements if e.type == 'character']
        if character_elements:
            prompt_parts.append(f"with {character_elements[0].description}")
        
        # Add objects and actions
        other_elements = [e for e in key_elements if e.type in ['object', 'action']]
        for element in other_elements[:2]:  # Limit to prevent overcrowding
            prompt_parts.append(element.description)
        
        # Add mood
        if mood != 'neutral':
            prompt_parts.append(f"{mood} atmosphere")
        
        # Combine into coherent prompt
        main_prompt = ', '.join(prompt_parts)
        
        # Add quality modifiers
        main_prompt += ", detailed, high quality, cinematic lighting"
        
        return main_prompt
    
    def _determine_generation_type(self, scene_type: str, elements: List[VisualElement]) -> str:
        """Determine whether to generate image or video"""
        if scene_type == 'dynamic':
            return 'video'
        elif scene_type == 'transition':
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
    
    def _generate_negative_prompt(self, scene_type: str, mood: str) -> str:
        """Generate negative prompt to avoid unwanted elements"""
        negative_elements = [
            "blurry", "low quality", "distorted", "ugly", "bad anatomy",
            "poorly drawn", "extra limbs", "duplicate", "mutation"
        ]
        
        if mood == 'peaceful':
            negative_elements.extend(["violence", "weapons", "blood", "war"])
        elif mood == 'bright':
            negative_elements.extend(["dark", "gloomy", "horror"])
        
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