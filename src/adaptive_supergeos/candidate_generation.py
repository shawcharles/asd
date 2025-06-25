"""Candidate supergeo generation for adaptive supergeo design.

This module provides methods for generating candidate supergeos from
geo embeddings, which can then be used in the optimization step.
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import itertools

from supergeos.supergeo_design import GeoUnit
from adaptive_supergeos.embedding import GeoEmbedding


class CandidateGenerator:
    """Base class for generating candidate supergeos."""
    
    def generate(self, geo_embeddings: List[GeoEmbedding]) -> List[List[GeoUnit]]:
        """Generate candidate supergeos from embeddings.
        
        Args:
            geo_embeddings: List of geographic units with embeddings
        
        Returns:
            List of candidate supergeos (each a list of GeoUnit objects)
        """
        raise NotImplementedError


class SimilarityGraphGenerator(CandidateGenerator):
    """Generate candidate supergeos based on similarity graph.
    
    Creates a similarity graph where nodes are geo units and edges
    represent similarity between embeddings. Then uses community
    detection and clustering to identify candidate supergeos.
    """
    
    def __init__(
        self,
        min_supergeo_size: int = 2,
        max_supergeo_size: int = 10,
        target_count: int = 100,
        similarity_threshold: float = 0.7,
    ):
        """Initialize the candidate generator.
        
        Args:
            min_supergeo_size: Minimum number of units in a supergeo
            max_supergeo_size: Maximum number of units in a supergeo
            target_count: Target number of candidates to generate
            similarity_threshold: Threshold for considering units similar
        """
        self.min_supergeo_size = max(min_supergeo_size, 2)
        self.max_supergeo_size = min(max_supergeo_size, 50)
        self.target_count = max(target_count, 10)
        self.similarity_threshold = max(0.1, min(similarity_threshold, 0.99))
    
    def _create_similarity_graph(
        self, 
        geo_embeddings: List[GeoEmbedding]
    ) -> nx.Graph:
        """Create a graph where nodes are geo units and edges represent similarity.
        
        Args:
            geo_embeddings: List of geographic units with embeddings
            
        Returns:
            NetworkX graph with nodes as indices into geo_embeddings
        """
        graph = nx.Graph()
        
        # Add nodes
        for i, emb in enumerate(geo_embeddings):
            graph.add_node(i, embedding=emb.embedding, geo_unit=emb.geo_unit)
        
        # Add edges based on embedding similarity
        for i in range(len(geo_embeddings)):
            for j in range(i+1, len(geo_embeddings)):
                emb_i = geo_embeddings[i].embedding
                emb_j = geo_embeddings[j].embedding
                
                # Calculate cosine similarity
                similarity = np.dot(emb_i, emb_j) / (
                    np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8
                )
                
                if similarity >= self.similarity_threshold:
                    graph.add_edge(i, j, weight=similarity)
                    
        return graph
    
    def _generate_community_candidates(
        self, 
        graph: nx.Graph, 
        geo_embeddings: List[GeoEmbedding]
    ) -> List[List[int]]:
        """Generate candidate supergeos using community detection.
        
        Args:
            graph: Similarity graph between geo units
            geo_embeddings: List of geographic units with embeddings
            
        Returns:
            List of candidate supergeos as lists of indices into geo_embeddings
        """
        candidates = []
        
        # Try different community detection methods
        try:
            # First try Louvain method for community detection
            communities = list(nx.community.louvain_communities(graph))
        except:
            try:
                # Fall back to greedy modularity communities
                communities = list(nx.community.greedy_modularity_communities(graph))
            except:
                # Last resort: connected components
                communities = list(nx.connected_components(graph))
        
        # Process each community
        for community in communities:
            community = list(community)
            
            # Skip if community is too small
            if len(community) < self.min_supergeo_size:
                continue
                
            # If community is too large, split it
            if len(community) > self.max_supergeo_size:
                # Create a subgraph for this community
                subgraph = graph.subgraph(community).copy()
                
                # Split using spectral clustering
                try:
                    # Number of clusters to create
                    n_clusters = max(2, len(community) // self.max_supergeo_size + 1)
                    
                    # Get the graph's Laplacian matrix for spectral clustering
                    laplacian = nx.normalized_laplacian_matrix(subgraph).todense()
                    eigvals, eigvecs = np.linalg.eigh(laplacian)
                    
                    # Use the eigenvectors corresponding to the smallest non-zero eigenvalues
                    # as features for K-means clustering
                    n_components = min(n_clusters, len(eigvals) - 1)
                    features = eigvecs[:, :n_components]
                    
                    # Apply K-means clustering
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(features)
                    
                    # Create subclusters
                    subclusters = [[] for _ in range(n_clusters)]
                    for i, label in enumerate(labels):
                        subclusters[label].append(community[i])
                    
                    # Add valid subclusters as candidates
                    for subcluster in subclusters:
                        if len(subcluster) >= self.min_supergeo_size:
                            candidates.append(subcluster)
                
                except:
                    # If spectral clustering fails, try a simpler approach
                    # Add the community in chunks of max_supergeo_size
                    for i in range(0, len(community), self.max_supergeo_size):
                        chunk = community[i:i+self.max_supergeo_size]
                        if len(chunk) >= self.min_supergeo_size:
                            candidates.append(chunk)
            else:
                # Add the whole community as a candidate
                candidates.append(community)
                
        return candidates
    
    def _generate_clique_candidates(
        self, 
        graph: nx.Graph, 
        geo_embeddings: List[GeoEmbedding]
    ) -> List[List[int]]:
        """Generate candidate supergeos using clique detection.
        
        Args:
            graph: Similarity graph between geo units
            geo_embeddings: List of geographic units with embeddings
            
        Returns:
            List of candidate supergeos as lists of indices into geo_embeddings
        """
        candidates = []
        
        # Find cliques in the graph within our size constraints
        try:
            # Look for cliques of size within our range
            for size in range(self.min_supergeo_size, self.max_supergeo_size + 1):
                # Find cliques of exact size (may be computationally expensive)
                cliques = list(nx.find_cliques(graph))
                for clique in cliques:
                    if self.min_supergeo_size <= len(clique) <= self.max_supergeo_size:
                        candidates.append(clique)
                        
                # Stop if we have enough candidates
                if len(candidates) >= self.target_count:
                    break
        except:
            pass  # Ignore errors in clique detection
            
        return candidates
    
    def _generate_expansion_candidates(
        self, 
        graph: nx.Graph, 
        geo_embeddings: List[GeoEmbedding],
        existing_candidates: List[List[int]]
    ) -> List[List[int]]:
        """Generate candidates by expanding existing candidates.
        
        Args:
            graph: Similarity graph between geo units
            geo_embeddings: List of geographic units with embeddings
            existing_candidates: Existing candidate supergeos
            
        Returns:
            List of new candidate supergeos
        """
        candidates = []
        seen = set(tuple(sorted(c)) for c in existing_candidates)
        
        # For each existing candidate
        for candidate in existing_candidates:
            if len(candidate) >= self.max_supergeo_size:
                continue
                
            # Find nodes connected to the candidate
            neighbors = set()
            for node in candidate:
                neighbors.update(graph.neighbors(node))
            neighbors -= set(candidate)  # Remove nodes already in candidate
            
            # Try adding each neighbor
            for neighbor in neighbors:
                new_candidate = candidate + [neighbor]
                new_candidate_tuple = tuple(sorted(new_candidate))
                
                if len(new_candidate) <= self.max_supergeo_size and new_candidate_tuple not in seen:
                    candidates.append(new_candidate)
                    seen.add(new_candidate_tuple)
                    
        return candidates
    
    def generate(self, geo_embeddings: List[GeoEmbedding]) -> List[List[GeoUnit]]:
        """Generate candidate supergeos from embeddings.
        
        Args:
            geo_embeddings: List of geographic units with embeddings
        
        Returns:
            List of candidate supergeos (each a list of GeoUnit objects)
        """
        # If we have too few geo units, just return a single candidate
        if len(geo_embeddings) < self.min_supergeo_size:
            return [[emb.geo_unit for emb in geo_embeddings]]
            
        # Create similarity graph
        graph = self._create_similarity_graph(geo_embeddings)
        
        # Generate candidates using different methods
        candidates_indices = []
        
        # Method 1: Community detection
        community_candidates = self._generate_community_candidates(graph, geo_embeddings)
        candidates_indices.extend(community_candidates)
        
        # Method 2: Clique detection (if we need more candidates)
        if len(candidates_indices) < self.target_count:
            clique_candidates = self._generate_clique_candidates(graph, geo_embeddings)
            candidates_indices.extend(clique_candidates)
            
        # Method 3: Expansion of existing candidates
        if len(candidates_indices) < self.target_count:
            expansion_candidates = self._generate_expansion_candidates(
                graph, geo_embeddings, candidates_indices
            )
            candidates_indices.extend(expansion_candidates)
            
        # Deduplicate candidates
        seen_candidates = set()
        unique_candidates = []
        
        for candidate in candidates_indices:
            candidate_tuple = tuple(sorted(candidate))
            if candidate_tuple not in seen_candidates:
                seen_candidates.add(candidate_tuple)
                unique_candidates.append(candidate)
                
        # Convert indices to GeoUnit objects
        result = []
        for candidate_indices in unique_candidates:
            geo_units = [geo_embeddings[i].geo_unit for i in candidate_indices]
            result.append(geo_units)
            
        # If we still need more candidates, create random groupings
        if len(result) < self.target_count:
            all_indices = list(range(len(geo_embeddings)))
            np.random.shuffle(all_indices)
            
            for i in range(0, len(all_indices), self.max_supergeo_size // 2):
                chunk = all_indices[i:i + self.max_supergeo_size // 2]
                if len(chunk) >= self.min_supergeo_size:
                    geo_units = [geo_embeddings[j].geo_unit for j in chunk]
                    result.append(geo_units)
                
        return result[:self.target_count]
