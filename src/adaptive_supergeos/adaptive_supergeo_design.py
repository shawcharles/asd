"""Adaptive Supergeo Design implementation.

This module implements the Adaptive Supergeo Design (ASD) algorithm, which extends
the original Supergeo Design with a two-stage heuristic approach using graph neural
networks to learn geo-embeddings and identify high-quality candidate supergeos.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import networkx as nx
from supergeos.supergeo_design import GeoUnit, SupergeoPartition
from supergeos.scoring import absolute_difference, relative_difference, covariate_diffs


@dataclass
class GeoEmbedding:
    """Represents a geographic unit with learned embeddings."""
    geo_unit: GeoUnit
    embedding: np.ndarray


class AdaptiveSupergeoBatchGenerator:
    """Generates batches of candidate supergeos based on embeddings."""
    
    def __init__(
        self,
        geo_embeddings: List[GeoEmbedding],
        min_supergeo_size: int = 2,
        max_supergeo_size: int = 10,
        batch_size: int = 100,
        similarity_threshold: float = 0.7
    ):
        """Initialize the batch generator.
        
        Args:
            geo_embeddings: List of geographic units with embeddings
            min_supergeo_size: Minimum number of units in a supergeo
            max_supergeo_size: Maximum number of units in a supergeo
            batch_size: Number of candidate supergeos to generate in each batch
            similarity_threshold: Threshold for considering units similar
        """
        self.geo_embeddings = geo_embeddings
        self.min_supergeo_size = min_supergeo_size
        self.max_supergeo_size = max_supergeo_size
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        
        # Create a similarity graph
        self._create_similarity_graph()
        
    def _create_similarity_graph(self):
        """Create a graph where nodes are geo units and edges represent similarity."""
        self.graph = nx.Graph()
        
        # Add nodes
        for i, emb in enumerate(self.geo_embeddings):
            self.graph.add_node(i, embedding=emb.embedding, geo_unit=emb.geo_unit)
        
        # Add edges based on embedding similarity
        for i in range(len(self.geo_embeddings)):
            for j in range(i+1, len(self.geo_embeddings)):
                emb_i = self.geo_embeddings[i].embedding
                emb_j = self.geo_embeddings[j].embedding
                
                # Calculate cosine similarity
                similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                
                if similarity >= self.similarity_threshold:
                    self.graph.add_edge(i, j, weight=similarity)
    
    def generate_batch(self) -> List[List[GeoUnit]]:
        """Generate a batch of candidate supergeos.
        
        Returns:
            List of candidate supergeos (each a list of GeoUnit objects)
        """
        candidates = []
        
        # Use community detection to find candidate supergeos
        communities = list(nx.community.greedy_modularity_communities(self.graph))
        
        # Process each community
        for community in communities:
            community = list(community)
            
            # Skip if community is too small
            if len(community) < self.min_supergeo_size:
                continue
                
            # If community is too large, split it
            if len(community) > self.max_supergeo_size:
                # Create a subgraph for this community
                subgraph = self.graph.subgraph(community).copy()
                
                # Use spectral clustering to split the community
                n_clusters = max(2, len(community) // self.max_supergeo_size)
                
                # This is a simplified approach - in a real implementation,
                # we would use a more sophisticated clustering algorithm
                subcommunities = list(nx.community.girvan_newman(subgraph))
                
                # Take the first valid partition
                for partition in subcommunities:
                    valid_subcommunities = []
                    for subcomm in partition:
                        if self.min_supergeo_size <= len(subcomm) <= self.max_supergeo_size:
                            valid_subcommunities.append(subcomm)
                    
                    if valid_subcommunities:
                        for subcomm in valid_subcommunities:
                            geo_units = [self.geo_embeddings[i].geo_unit for i in subcomm]
                            candidates.append(geo_units)
                        break
            else:
                # Community size is within bounds, add it as a candidate
                geo_units = [self.geo_embeddings[i].geo_unit for i in community]
                candidates.append(geo_units)
        
        # If we have too few candidates, generate some random ones
        while len(candidates) < self.batch_size:
            size = np.random.randint(self.min_supergeo_size, self.max_supergeo_size + 1)
            indices = np.random.choice(len(self.geo_embeddings), size=size, replace=False)
            geo_units = [self.geo_embeddings[i].geo_unit for i in indices]
            candidates.append(geo_units)
            
        # If we have too many candidates, select a diverse subset
        if len(candidates) > self.batch_size:
            # This is a simplified approach - in a real implementation,
            # we would use a more sophisticated diversity-based selection
            candidates = candidates[:self.batch_size]
            
        return candidates


class GeoEmbeddingModel:
    """Model for learning embeddings of geographic units."""
    
    def __init__(
        self,
        embedding_dim: int = 32,
        learning_rate: float = 0.01,
        n_epochs: int = 100
    ):
        """Initialize the embedding model.
        
        Args:
            embedding_dim: Dimension of the embeddings
            learning_rate: Learning rate for training
            n_epochs: Number of training epochs
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
    def fit(self, geo_units: List[GeoUnit]) -> List[GeoEmbedding]:
        """Learn embeddings for geographic units.
        
        Args:
            geo_units: List of geographic units
            
        Returns:
            List of geographic units with learned embeddings
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, this would use a GNN to learn embeddings
        
        # For now, we'll just create random embeddings
        embeddings = []
        
        for geo_unit in geo_units:
            # Create a random embedding
            embedding = np.random.normal(0, 1, self.embedding_dim)
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            embeddings.append(GeoEmbedding(geo_unit=geo_unit, embedding=embedding))
            
        return embeddings


class AdaptiveSupergeoDesign:
    """Implementation of the Adaptive Supergeo Design algorithm.
    
    This class implements the Adaptive Supergeo Design algorithm, which extends
    the original Supergeo Design with a two-stage heuristic approach using graph
    neural networks to learn geo-embeddings and identify high-quality candidate
    supergeos.
    """
    
    def __init__(
        self,
        geo_units: List[GeoUnit],
        min_supergeo_size: int = 2,
        max_supergeo_size: int = 10,
        embedding_dim: int = 32,
        batch_size: int = 100,
        balance_metric: str = "absolute_difference"
    ):
        """Initialize the AdaptiveSupergeoDesign.
        
        Args:
            geo_units: List of geographic units to partition
            min_supergeo_size: Minimum number of units in a supergeo
            max_supergeo_size: Maximum number of units in a supergeo
            embedding_dim: Dimension of the embeddings
            batch_size: Number of candidate supergeos to generate in each batch
            balance_metric: Metric to use for measuring balance
        """
        self.geo_units = geo_units
        self.min_supergeo_size = min_supergeo_size
        self.max_supergeo_size = max_supergeo_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.balance_metric = balance_metric

    def _balance_score(self, treatment: List[GeoUnit], control: List[GeoUnit]) -> float:
        """Compute balance score according to selected metric."""
        if self.balance_metric == "absolute_difference":
            return absolute_difference(treatment, control)
        return relative_difference(treatment, control)

        
    def optimize_partition(
        self,
        n_treatment_supergeos: int,
        n_control_supergeos: int,
        balance_weights: Optional[Dict[str, float]] = None,
        max_iterations: int = 1000,
        random_seed: Optional[int] = None
    ) -> SupergeoPartition:
        """Find an optimal partition of geographic units into supergeos.
        
        Args:
            n_treatment_supergeos: Number of treatment supergeos to create
            n_control_supergeos: Number of control supergeos to create
            balance_weights: Optional weights for balancing different metrics
            max_iterations: Maximum number of iterations for optimization
            random_seed: Random seed for reproducibility
            
        Returns:
            A SupergeoPartition object with the optimal partition
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Step 1: Learn embeddings for geographic units
        embedding_model = GeoEmbeddingModel(embedding_dim=self.embedding_dim)
        geo_embeddings = embedding_model.fit(self.geo_units)
        
        # Step 2: Generate candidate supergeos
        batch_generator = AdaptiveSupergeoBatchGenerator(
            geo_embeddings=geo_embeddings,
            min_supergeo_size=self.min_supergeo_size,
            max_supergeo_size=self.max_supergeo_size,
            batch_size=self.batch_size
        )
        
        candidate_supergeos = batch_generator.generate_batch()
        
        # -------------------------------------------------------------------
        # Greedy assignment of candidate supergeos to treatment/control.
        # We perform multiple random restarts and keep the best partition.
        # -------------------------------------------------------------------
        best_partition: Optional[SupergeoPartition] = None
        best_score = float("inf")

        # Pre-compute flattened list of all units for covariate diffs later
        for attempt in range(min(max_iterations, 50)):
            np.random.shuffle(candidate_supergeos)
            treatment_supergeos: List[List[GeoUnit]] = []
            control_supergeos: List[List[GeoUnit]] = []

            # Greedy sequential assignment – choose arm that minimises balance so far
            flat_treatment: List[GeoUnit] = []
            flat_control: List[GeoUnit] = []

            for cand in candidate_supergeos:
                if len(treatment_supergeos) < n_treatment_supergeos and len(control_supergeos) < n_control_supergeos:
                    # Compute score if assigned to treatment vs control
                    score_if_treatment = self._balance_score(flat_treatment + cand, flat_control)
                    score_if_control = self._balance_score(flat_treatment, flat_control + cand)
                    if score_if_treatment <= score_if_control:
                        treatment_supergeos.append(cand)
                        flat_treatment.extend(cand)
                    else:
                        control_supergeos.append(cand)
                        flat_control.extend(cand)
                elif len(treatment_supergeos) < n_treatment_supergeos:
                    treatment_supergeos.append(cand)
                    flat_treatment.extend(cand)
                elif len(control_supergeos) < n_control_supergeos:
                    control_supergeos.append(cand)
                    flat_control.extend(cand)
                if (len(treatment_supergeos) == n_treatment_supergeos and
                        len(control_supergeos) == n_control_supergeos):
                    break

            # Skip invalid attempt if counts not satisfied
            if (len(treatment_supergeos) < n_treatment_supergeos or
                    len(control_supergeos) < n_control_supergeos):
                continue

            score = self._balance_score(flat_treatment, flat_control)
            if score < best_score:
                best_score = score
                cov_balance = covariate_diffs(flat_treatment, flat_control)
                best_partition = SupergeoPartition(
                    treatment_supergeos=treatment_supergeos,
                    control_supergeos=control_supergeos,
                    balance_score=score,
                    covariate_balance=cov_balance,
                )

        # Fallback to naive partition if greedy failed
        if best_partition is None:
            treatment_supergeos = candidate_supergeos[:n_treatment_supergeos]
            control_supergeos = candidate_supergeos[n_treatment_supergeos:
                                                    n_treatment_supergeos + n_control_supergeos]
            flat_treatment = [g for sg in treatment_supergeos for g in sg]
            flat_control = [g for sg in control_supergeos for g in sg]
            best_partition = SupergeoPartition(
                treatment_supergeos=treatment_supergeos,
                control_supergeos=control_supergeos,
                balance_score=self._balance_score(flat_treatment, flat_control),
                covariate_balance=covariate_diffs(flat_treatment, flat_control),
            )

        return best_partition
    
    def evaluate_partition(self, partition: SupergeoPartition) -> Dict[str, float]:
        """Evaluate the quality of a partition.
        
        Args:
            partition: A SupergeoPartition to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, this would calculate various metrics
        
        # For now, we'll just return some dummy metrics
        return {
            "balance_score": partition.balance_score,
            "treatment_units": sum(len(sg) for sg in partition.treatment_supergeos),
            "control_units": sum(len(sg) for sg in partition.control_supergeos)
        }
