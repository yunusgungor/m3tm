"""
Data Governance Engine
======================

DataHub-integrated data governance, lineage tracking, and metadata management.

Context7 References:
- /datahub/datahub: Metadata management and data catalog
- /open-policy-agent/opa: Policy engine for data governance rules
- /apache/kafka: Event streaming for lineage tracking
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class LineageDirection(Enum):
    """Data lineage direction"""
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    BOTH = "both"

@dataclass
class DataAsset:
    """Data asset metadata container"""
    asset_id: str
    name: str
    description: Optional[str] = None
    classification: DataClassification = DataClassification.INTERNAL
    owner: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class LineageRelation:
    """Data lineage relationship"""
    source_asset: str
    target_asset: str
    relation_type: str  # "DERIVES_FROM", "PRODUCES", "CONSUMES"
    transformation_info: Optional[Dict[str, Any]] = None
    confidence_score: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class GovernanceEngine:
    """
    Enterprise data governance engine with DataHub integration.
    
    Features:
    - Data asset registration and metadata management
    - Data lineage tracking and visualization
    - Classification and tagging
    - Policy enforcement integration
    - Quality monitoring
    
    Context7 Pattern:
    - Uses DataHub GraphQL API for metadata operations
    - Integrates with OPA for policy evaluation
    - Event-driven lineage tracking via Kafka
    """
    
    def __init__(self, 
                 datahub_endpoint: str = "http://localhost:8080",
                 datahub_token: Optional[str] = None,
                 opa_endpoint: str = "http://localhost:8181",
                 kafka_config: Optional[Dict[str, Any]] = None):
        """
        Initialize governance engine with external integrations.
        
        Args:
            datahub_endpoint: DataHub GMS endpoint URL
            datahub_token: Authentication token for DataHub
            opa_endpoint: Open Policy Agent endpoint
            kafka_config: Kafka configuration for lineage events
        """
        self.datahub_endpoint = datahub_endpoint
        self.datahub_token = datahub_token
        self.opa_endpoint = opa_endpoint
        self.kafka_config = kafka_config or {}
        
        # Internal storage for metadata (fallback when DataHub unavailable)
        self._assets: Dict[str, DataAsset] = {}
        self._lineage: List[LineageRelation] = []
        
        # Policy cache
        self._policy_cache: Dict[str, Any] = {}
        
        logger.info(f"GovernanceEngine initialized with DataHub: {datahub_endpoint}")
    
    async def register_asset(self, asset: DataAsset) -> bool:
        """
        Register a data asset with metadata management.
        
        Context7 Pattern: Uses DataHub metadata ingestion API
        """
        try:
            # Create DataHub dataset metadata
            dataset_metadata = {
                "urn": f"urn:li:dataset:(urn:li:dataPlatform:m3tm,{asset.asset_id},PROD)",
                "properties": {
                    "name": asset.name,
                    "description": asset.description,
                    "customProperties": asset.properties
                },
                "ownership": {
                    "owners": [{"owner": asset.owner, "type": "DATAOWNER"}] if asset.owner else []
                },
                "tags": [{"tag": tag} for tag in asset.tags],
                "globalTags": {
                    "tags": [
                        {"tag": f"urn:li:tag:{asset.classification.value}"},
                        {"tag": f"urn:li:tag:m3tm-asset"}
                    ]
                }
            }
            
            # TODO: Integrate with actual DataHub GraphQL API
            # For now, store locally
            self._assets[asset.asset_id] = asset
            
            await self._emit_governance_event("asset_registered", {
                "asset_id": asset.asset_id,
                "classification": asset.classification.value,
                "timestamp": asset.created_at.isoformat()
            })
            
            logger.info(f"Data asset registered: {asset.asset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register asset {asset.asset_id}: {e}")
            return False
    
    async def track_lineage(self, relation: LineageRelation) -> bool:
        """
        Track data lineage relationship.
        
        Context7 Pattern: Uses DataHub lineage API and Kafka events
        """
        try:
            # Create DataHub lineage metadata
            lineage_metadata = {
                "upstreamLineage": {
                    "upstreams": [{
                        "dataset": f"urn:li:dataset:(urn:li:dataPlatform:m3tm,{relation.source_asset},PROD)",
                        "type": relation.relation_type.upper()
                    }]
                }
            }
            
            # Store lineage relation
            self._lineage.append(relation)
            
            # Emit lineage event for real-time tracking
            await self._emit_lineage_event(relation)
            
            logger.info(f"Lineage tracked: {relation.source_asset} -> {relation.target_asset}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track lineage: {e}")
            return False
    
    async def get_lineage(self, asset_id: str, 
                         direction: LineageDirection = LineageDirection.BOTH,
                         depth: int = 3) -> Dict[str, Any]:
        """
        Get data lineage for an asset.
        
        Returns lineage graph with upstream/downstream relationships.
        """
        try:
            lineage_graph = {
                "asset_id": asset_id,
                "direction": direction.value,
                "depth": depth,
                "nodes": {},
                "edges": []
            }
            
            # Build lineage graph from stored relations
            visited = set()
            queue = [(asset_id, 0)]
            
            while queue:
                current_asset, current_depth = queue.pop(0)
                if current_asset in visited or current_depth >= depth:
                    continue
                    
                visited.add(current_asset)
                
                # Add asset node
                if current_asset in self._assets:
                    lineage_graph["nodes"][current_asset] = {
                        "name": self._assets[current_asset].name,
                        "classification": self._assets[current_asset].classification.value,
                        "depth": current_depth
                    }
                
                # Find related assets
                for relation in self._lineage:
                    if direction in [LineageDirection.UPSTREAM, LineageDirection.BOTH]:
                        if relation.target_asset == current_asset:
                            lineage_graph["edges"].append({
                                "source": relation.source_asset,
                                "target": relation.target_asset,
                                "type": relation.relation_type,
                                "confidence": relation.confidence_score
                            })
                            queue.append((relation.source_asset, current_depth + 1))
                    
                    if direction in [LineageDirection.DOWNSTREAM, LineageDirection.BOTH]:
                        if relation.source_asset == current_asset:
                            lineage_graph["edges"].append({
                                "source": relation.source_asset,
                                "target": relation.target_asset,
                                "type": relation.relation_type,
                                "confidence": relation.confidence_score
                            })
                            queue.append((relation.target_asset, current_depth + 1))
            
            return lineage_graph
            
        except Exception as e:
            logger.error(f"Failed to get lineage for {asset_id}: {e}")
            return {"error": str(e)}
    
    async def evaluate_policy(self, asset_id: str, action: str, 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate governance policy using OPA.
        
        Context7 Pattern: Uses OPA REST API for policy evaluation
        """
        try:
            policy_input = {
                "input": {
                    "asset_id": asset_id,
                    "action": action,
                    "context": context or {},
                    "asset_metadata": self._assets.get(asset_id, {})
                }
            }
            
            # TODO: Integrate with actual OPA API
            # For now, return basic policy evaluation
            asset = self._assets.get(asset_id)
            if not asset:
                return {"allow": False, "reason": "Asset not found"}
            
            # Basic classification-based policy
            if asset.classification == DataClassification.RESTRICTED and action == "read":
                return {
                    "allow": False,
                    "reason": "Restricted data requires special authorization",
                    "required_permissions": ["data.restricted.read"]
                }
            
            return {"allow": True, "reason": "Policy evaluation passed"}
            
        except Exception as e:
            logger.error(f"Policy evaluation failed for {asset_id}: {e}")
            return {"allow": False, "reason": f"Policy evaluation error: {e}"}
    
    async def get_asset_metadata(self, asset_id: str) -> Optional[DataAsset]:
        """Get asset metadata by ID."""
        return self._assets.get(asset_id)
    
    async def search_assets(self, query: str, 
                          classification: Optional[DataClassification] = None,
                          tags: Optional[List[str]] = None) -> List[DataAsset]:
        """
        Search assets by query, classification, and tags.
        
        Context7 Pattern: Uses DataHub search API with ElasticSearch
        """
        try:
            results = []
            
            for asset in self._assets.values():
                # Text search in name and description
                if query.lower() in asset.name.lower():
                    match = True
                elif asset.description and query.lower() in asset.description.lower():
                    match = True
                else:
                    match = False
                
                # Classification filter
                if classification and asset.classification != classification:
                    match = False
                
                # Tags filter
                if tags and not any(tag in asset.tags for tag in tags):
                    match = False
                
                if match:
                    results.append(asset)
            
            return results
            
        except Exception as e:
            logger.error(f"Asset search failed: {e}")
            return []
    
    async def _emit_governance_event(self, event_type: str, payload: Dict[str, Any]):
        """Emit governance event to Kafka for real-time monitoring."""
        try:
            event = {
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
                "source": "m3tm-governance"
            }
            
            # TODO: Integrate with actual Kafka producer
            logger.debug(f"Governance event: {json.dumps(event)}")
            
        except Exception as e:
            logger.error(f"Failed to emit governance event: {e}")
    
    async def _emit_lineage_event(self, relation: LineageRelation):
        """Emit lineage event for real-time tracking."""
        try:
            event = {
                "event_type": "lineage_tracked",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "source_asset": relation.source_asset,
                    "target_asset": relation.target_asset,
                    "relation_type": relation.relation_type,
                    "confidence_score": relation.confidence_score
                },
                "source": "m3tm-lineage"
            }
            
            # TODO: Integrate with actual Kafka producer
            logger.debug(f"Lineage event: {json.dumps(event)}")
            
        except Exception as e:
            logger.error(f"Failed to emit lineage event: {e}")
    
    async def get_governance_metrics(self) -> Dict[str, Any]:
        """Get governance metrics and statistics."""
        try:
            total_assets = len(self._assets)
            classification_counts = {}
            tag_counts = {}
            
            for asset in self._assets.values():
                # Classification distribution
                cls = asset.classification.value
                classification_counts[cls] = classification_counts.get(cls, 0) + 1
                
                # Tag distribution
                for tag in asset.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return {
                "total_assets": total_assets,
                "total_lineage_relations": len(self._lineage),
                "classification_distribution": classification_counts,
                "popular_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                "governance_health_score": min(100, total_assets * 10),  # Simple scoring
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get governance metrics: {e}")
            return {"error": str(e)}

# Export main class
__all__ = ["GovernanceEngine", "DataAsset", "LineageRelation", "DataClassification", "LineageDirection"]
