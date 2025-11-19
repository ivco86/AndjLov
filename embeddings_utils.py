"""
CLIP Embeddings and Semantic Search Utilities
Natural language image search using OpenAI's CLIP model
"""

import os
import numpy as np
from PIL import Image

# Try to import transformers and torch
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("Warning: transformers/torch not installed. Semantic search will be disabled.")
    print("Install with: pip install transformers torch")


class CLIPEmbeddingsGenerator:
    """Singleton class for managing CLIP model and generating embeddings"""

    _instance = None
    _model = None
    _processor = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize CLIP model (lazy loading)"""
        if not HAS_CLIP:
            self.available = False
            return

        self.available = True

        # Only load model once
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load CLIP model and processor"""
        try:
            print("Loading CLIP model (this may take a while on first run)...")

            # Use CPU or GPU
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self._device}")

            # Load OpenAI CLIP model (ViT-B/32 is a good balance of speed and quality)
            model_name = "openai/clip-vit-base-patch32"
            self._model = CLIPModel.from_pretrained(model_name).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(model_name)

            print("✅ CLIP model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading CLIP model: {e}")
            self.available = False
            self._model = None
            self._processor = None

    def generate_image_embedding(self, image_path):
        """
        Generate CLIP embedding for an image

        Args:
            image_path: Path to the image file

        Returns:
            numpy.ndarray: 512-dimensional embedding vector, or None if failed
        """
        if not self.available or self._model is None:
            return None

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)

            # Generate embedding
            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)

            # Normalize embedding (for cosine similarity)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy array
            embedding = image_features.cpu().numpy().flatten()

            return embedding

        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}")
            return None

    def generate_text_embedding(self, text_query):
        """
        Generate CLIP embedding for a text query

        Args:
            text_query: Natural language search query

        Returns:
            numpy.ndarray: 512-dimensional embedding vector, or None if failed
        """
        if not self.available or self._model is None:
            return None

        try:
            # Preprocess text
            inputs = self._processor(text=[text_query], return_tensors="pt", padding=True).to(self._device)

            # Generate embedding
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)

            # Normalize embedding (for cosine similarity)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to numpy array
            embedding = text_features.cpu().numpy().flatten()

            return embedding

        except Exception as e:
            print(f"Error generating text embedding for '{text_query}': {e}")
            return None

    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector (numpy array)
            embedding2: Second embedding vector (numpy array)

        Returns:
            float: Cosine similarity score (0-1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        try:
            # Cosine similarity (embeddings are already normalized)
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)

        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0


# Global instance
_clip_generator = None


def get_clip_generator():
    """Get or create the global CLIP generator instance"""
    global _clip_generator
    if _clip_generator is None:
        _clip_generator = CLIPEmbeddingsGenerator()
    return _clip_generator


def is_clip_available():
    """Check if CLIP model is available"""
    if not HAS_CLIP:
        return False

    generator = get_clip_generator()
    return generator.available


def generate_embedding_for_image(image_path):
    """
    Generate CLIP embedding for an image

    Args:
        image_path: Path to the image file

    Returns:
        list: 512-dimensional embedding as list of floats, or None
    """
    generator = get_clip_generator()

    if not generator.available:
        return None

    embedding = generator.generate_image_embedding(image_path)

    if embedding is not None:
        return embedding.tolist()

    return None


def search_by_text_query(text_query, image_embeddings):
    """
    Search images by natural language query

    Args:
        text_query: Natural language search query (e.g., "sunset over ocean")
        image_embeddings: List of dicts with 'id' and 'embedding' keys

    Returns:
        list: Sorted list of {'image_id': int, 'similarity': float}, or None
    """
    generator = get_clip_generator()

    if not generator.available:
        return None

    # Generate text embedding
    text_embedding = generator.generate_text_embedding(text_query)

    if text_embedding is None:
        return None

    # Compute similarities
    results = []

    for item in image_embeddings:
        image_id = item['id']
        image_embedding = np.array(item['embedding'])

        similarity = generator.compute_similarity(text_embedding, image_embedding)

        results.append({
            'image_id': image_id,
            'similarity': similarity
        })

    # Sort by similarity (descending)
    results.sort(key=lambda x: x['similarity'], reverse=True)

    return results


def find_similar_images(query_embedding, all_embeddings, top_k=10, exclude_id=None):
    """
    Find most similar images to a query embedding

    Args:
        query_embedding: Query embedding (list or numpy array)
        all_embeddings: List of dicts with 'id' and 'embedding' keys
        top_k: Number of results to return
        exclude_id: Image ID to exclude from results (e.g., the query image itself)

    Returns:
        list: Top K similar images [{'image_id': int, 'similarity': float}, ...]
    """
    generator = get_clip_generator()

    if not generator.available:
        return []

    query_emb = np.array(query_embedding)

    results = []

    for item in all_embeddings:
        image_id = item['id']

        # Skip excluded image
        if exclude_id is not None and image_id == exclude_id:
            continue

        image_embedding = np.array(item['embedding'])

        similarity = generator.compute_similarity(query_emb, image_embedding)

        results.append({
            'image_id': image_id,
            'similarity': similarity
        })

    # Sort by similarity (descending)
    results.sort(key=lambda x: x['similarity'], reverse=True)

    # Return top K
    return results[:top_k]


def embedding_to_blob(embedding):
    """
    Convert embedding numpy array to binary blob for database storage

    Args:
        embedding: numpy array or list

    Returns:
        bytes: Binary representation
    """
    if embedding is None:
        return None

    if isinstance(embedding, list):
        embedding = np.array(embedding, dtype=np.float32)

    return embedding.tobytes()


def blob_to_embedding(blob):
    """
    Convert binary blob from database back to numpy array

    Args:
        blob: Binary blob from database

    Returns:
        numpy.ndarray: Embedding vector
    """
    if blob is None:
        return None

    return np.frombuffer(blob, dtype=np.float32)
