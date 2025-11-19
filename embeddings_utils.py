"""
Embeddings generation utilities for AI Gallery
Generates vector embeddings for semantic search using CLIP or similar models
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import pickle


class EmbeddingsGenerator:
    """Generate and search image embeddings for semantic search"""

    def __init__(self, model_name: str = 'clip'):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._init_model()

    def _init_model(self):
        """Initialize the embedding model (CLIP)"""
        try:
            # Try to import transformers (HuggingFace)
            from transformers import CLIPProcessor, CLIPModel
            import torch

            print(f"Loading {self.model_name} model...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")

        except ImportError:
            print("⚠️ Warning: transformers library not installed.")
            print("To enable semantic search, install: pip install transformers torch pillow")
            print("Embeddings functionality will be disabled until installed.")
            self.model = None

        except Exception as e:
            print(f"⚠️ Warning: Could not load CLIP model: {e}")
            print("Embeddings functionality will be disabled.")
            self.model = None

    def is_available(self) -> bool:
        """Check if embedding generation is available"""
        return self.model is not None

    def generate_image_embedding(self, image_path: str) -> Optional[bytes]:
        """
        Generate embedding vector for an image

        Returns:
            bytes: Serialized numpy array (pickle), or None if model unavailable
        """
        if not self.is_available():
            return None

        try:
            from PIL import Image
            import torch

            # Load and process image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Normalize and convert to numpy
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding_array = image_features.cpu().numpy()[0]

            # Serialize to bytes for storage in SQLite BLOB
            return pickle.dumps(embedding_array)

        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}")
            return None

    def generate_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding vector for text query

        Returns:
            numpy array or None if model unavailable
        """
        if not self.is_available():
            return None

        try:
            import torch

            # Process text
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            # Normalize and convert to numpy
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embedding_array = text_features.cpu().numpy()[0]

            return embedding_array

        except Exception as e:
            print(f"Error generating text embedding for '{text}': {e}")
            return None

    def deserialize_embedding(self, embedding_blob: bytes) -> np.ndarray:
        """Deserialize embedding from database BLOB"""
        return pickle.loads(embedding_blob)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def search_similar_images(self,
                             query_embedding: np.ndarray,
                             all_embeddings: List[Tuple[int, bytes]],
                             top_k: int = 10,
                             threshold: float = 0.0) -> List[Tuple[int, float]]:
        """
        Search for similar images using cosine similarity

        Args:
            query_embedding: Query vector (numpy array)
            all_embeddings: List of (image_id, embedding_blob) tuples
            top_k: Number of top results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of (image_id, similarity_score) tuples, sorted by score descending
        """
        if not self.is_available():
            return []

        results = []

        for image_id, embedding_blob in all_embeddings:
            try:
                image_embedding = self.deserialize_embedding(embedding_blob)
                similarity = self.cosine_similarity(query_embedding, image_embedding)

                if similarity >= threshold:
                    results.append((image_id, similarity))

            except Exception as e:
                print(f"Error comparing embedding for image {image_id}: {e}")
                continue

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def batch_generate_embeddings(self, image_paths: List[str]) -> List[Tuple[str, Optional[bytes]]]:
        """
        Generate embeddings for multiple images (can be optimized for batch processing)

        Returns:
            List of (image_path, embedding_blob) tuples
        """
        results = []

        for image_path in image_paths:
            embedding = self.generate_image_embedding(image_path)
            results.append((image_path, embedding))

        return results


# Global instance (singleton pattern)
_embeddings_generator = None


def get_embeddings_generator() -> EmbeddingsGenerator:
    """Get or create the global embeddings generator instance"""
    global _embeddings_generator

    if _embeddings_generator is None:
        _embeddings_generator = EmbeddingsGenerator()

    return _embeddings_generator


def search_by_text(db, query_text: str, top_k: int = 20) -> List[dict]:
    """
    High-level function to search images by text query

    Args:
        db: Database instance
        query_text: Natural language query (e.g., "a dog playing in the park")
        top_k: Number of results to return

    Returns:
        List of image dicts with similarity scores
    """
    generator = get_embeddings_generator()

    if not generator.is_available():
        print("⚠️ Semantic search unavailable - embeddings model not loaded")
        return []

    # Generate query embedding
    query_embedding = generator.generate_text_embedding(query_text)
    if query_embedding is None:
        return []

    # Get all image embeddings from database
    all_embeddings = db.get_all_embeddings(model_name='clip-vit-base-patch32')

    if not all_embeddings:
        print("No image embeddings found in database. Generate embeddings first.")
        return []

    # Convert to format for search
    embeddings_data = [(e['image_id'], e['vector']) for e in all_embeddings]

    # Search for similar images
    similar_images = generator.search_similar_images(query_embedding, embeddings_data, top_k=top_k)

    # Fetch full image data
    results = []
    for image_id, similarity_score in similar_images:
        image = db.get_image(image_id)
        if image:
            image['similarity_score'] = similarity_score
            results.append(image)

    return results


def search_by_image(db, image_id: int, top_k: int = 20) -> List[dict]:
    """
    High-level function to search for similar images by image

    Args:
        db: Database instance
        image_id: ID of the query image
        top_k: Number of results to return

    Returns:
        List of similar image dicts with similarity scores
    """
    generator = get_embeddings_generator()

    if not generator.is_available():
        print("⚠️ Semantic search unavailable - embeddings model not loaded")
        return []

    # Get query image embedding
    query_embedding_data = db.get_embedding(image_id)
    if not query_embedding_data:
        print(f"No embedding found for image {image_id}")
        return []

    query_embedding = generator.deserialize_embedding(query_embedding_data['vector'])

    # Get all image embeddings from database
    all_embeddings = db.get_all_embeddings(model_name='clip-vit-base-patch32')

    if not all_embeddings:
        return []

    # Convert to format for search (exclude the query image itself)
    embeddings_data = [
        (e['image_id'], e['vector'])
        for e in all_embeddings
        if e['image_id'] != image_id
    ]

    # Search for similar images
    similar_images = generator.search_similar_images(query_embedding, embeddings_data, top_k=top_k)

    # Fetch full image data
    results = []
    for img_id, similarity_score in similar_images:
        image = db.get_image(img_id)
        if image:
            image['similarity_score'] = similarity_score
            results.append(image)

    return results
