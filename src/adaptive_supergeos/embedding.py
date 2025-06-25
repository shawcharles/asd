"""Embedding models for adaptive supergeo design.

This module provides embedding methods for geographic units
that capture their underlying similarity structure for partitioning.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple
import networkx as nx
from dataclasses import dataclass

from supergeos.supergeo_design import GeoUnit


@dataclass
class GeoEmbedding:
    """Geographic unit with learned embedding vector."""
    geo_unit: GeoUnit
    embedding: np.ndarray


class BaseEmbeddingModel:
    """Base class for geo-unit embedding models."""
    
    def fit(self, geo_units: List[GeoUnit]) -> List[GeoEmbedding]:
        """Learn embeddings for geographic units.
        
        Args:
            geo_units: List of geographic units
            
        Returns:
            List of geographic units with learned embeddings
        """
        raise NotImplementedError


class SimilarityEmbeddingModel(BaseEmbeddingModel):
    """Simple embedding model based on feature similarity.
    
    This model creates a similarity graph between geo units based on
    response/spend ratios and covariate similarity, then performs
    spectral embedding on this graph.
    """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        ratio_weight: float = 1.0,
        covariate_weight: float = 1.0,
    ):
        """Initialize the model.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            ratio_weight: Weight of response/spend ratio in similarity calculation
            covariate_weight: Weight of covariate similarity
        """
        self.embedding_dim = min(embedding_dim, 128)  # Cap dimension
        self.ratio_weight = ratio_weight
        self.covariate_weight = covariate_weight
        
    def _create_similarity_graph(self, geo_units: List[GeoUnit]) -> nx.Graph:
        """Create a similarity graph between geo units.
        
        Args:
            geo_units: List of geographic units
            
        Returns:
            NetworkX graph with similarity edges
        """
        G = nx.Graph()
        
        # Add nodes
        for i, unit in enumerate(geo_units):
            G.add_node(i, geo_unit=unit)
        
        # Helper: Calculate ratio safely
        def calc_ratio(g: GeoUnit) -> float:
            return g.response / g.spend if abs(g.spend) > 1e-9 else g.response

        # Precompute all ratios
        ratios = [calc_ratio(g) for g in geo_units]
        
        # Normalize ratios to [0,1] scale
        ratio_min = min(ratios)
        ratio_max = max(ratios)
        ratio_range = max(ratio_max - ratio_min, 1e-9)
        norm_ratios = [(r - ratio_min) / ratio_range for r in ratios]
        
        # Precompute all covariate sets (for faster comparison)
        all_covars = set()
        for g in geo_units:
            if g.covariates:
                all_covars.update(g.covariates.keys())
                
        # Add weighted edges based on similarity
        for i in range(len(geo_units)):
            for j in range(i+1, len(geo_units)):
                # 1. Ratio similarity (scaled by ratio_weight)
                ratio_sim = 1 - abs(norm_ratios[i] - norm_ratios[j])
                
                # 2. Covariate similarity (if available)
                cov_sim = 0.0
                if geo_units[i].covariates and geo_units[j].covariates:
                    matches = 0
                    total = len(all_covars)
                    for k in all_covars:
                        v_i = geo_units[i].covariates.get(k, 0.0)
                        v_j = geo_units[j].covariates.get(k, 0.0)
                        # Simple strategy: similar if within 10% of each other
                        if abs(v_i - v_j) / (max(abs(v_i), abs(v_j), 1e-9)) < 0.1:
                            matches += 1
                    cov_sim = matches / total if total > 0 else 0.0
                
                # Weighted combination
                final_sim = (ratio_sim * self.ratio_weight + 
                             cov_sim * self.covariate_weight) / (
                                 self.ratio_weight + self.covariate_weight)
                
                # Only add edges with non-zero similarity
                if final_sim > 0.1:  # Threshold for edge creation
                    G.add_edge(i, j, weight=final_sim)
        
        return G
    
    def fit(self, geo_units: List[GeoUnit]) -> List[GeoEmbedding]:
        """Learn embeddings via spectral embedding on similarity graph.
        
        Args:
            geo_units: List of geographic units
            
        Returns:
            List of geographic units with learned embeddings
        """
        # Default to random embeddings if only a few units
        if len(geo_units) < 2 * self.embedding_dim:
            # Just use random embeddings - too few for meaningful spectral embedding
            embeddings = np.random.normal(0, 1, (len(geo_units), self.embedding_dim))
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            return [
                GeoEmbedding(geo_unit=g, embedding=e) 
                for g, e in zip(geo_units, embeddings)
            ]
        
        # Create similarity graph
        G = self._create_similarity_graph(geo_units)
        
        # If graph is disconnected, ensure at least minimal connectivity
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            if len(components) > 1:
                # Add edges between components to make graph connected
                for i in range(len(components) - 1):
                    G.add_edge(
                        list(components[i])[0],
                        list(components[i+1])[0],
                        weight=0.1  # Low weight connection
                    )
        
        # Use spectral embedding
        dim = min(self.embedding_dim, len(geo_units) - 1, 128)
        embeddings = nx.spectral_embedding(G, dim)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1e-8  # Avoid division by zero
        embeddings = embeddings / norms
        
        # Create GeoEmbedding objects
        result = []
        for i, geo_unit in enumerate(geo_units):
            result.append(GeoEmbedding(
                geo_unit=geo_unit,
                embedding=embeddings[i]
            ))
            
        return result
                

class RandomEmbeddingModel(BaseEmbeddingModel):
    """Random embedding model for testing and baseline comparison."""
    
    def __init__(self, embedding_dim: int = 32):
        """Initialize the random embedding model.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
        """
        self.embedding_dim = embedding_dim
        
    def fit(self, geo_units: List[GeoUnit]) -> List[GeoEmbedding]:
        """Generate random embeddings for geographic units.
        
        Args:
            geo_units: List of geographic units
            
        Returns:
            List of geographic units with random embeddings
        """
        result = []
        for geo_unit in geo_units:
            # Create random embedding
            embedding = np.random.normal(0, 1, self.embedding_dim)
            # Normalize to unit length
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            result.append(GeoEmbedding(geo_unit=geo_unit, embedding=embedding))
            
        return result
