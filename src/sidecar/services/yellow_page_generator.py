"""Yellow Page Generator - Scans expert principles and generates registry.

This module is responsible for:
1. Scanning experts/ directory at startup
2. Extracting metadata from each expert's principles.md
3. Generating yellow_page.generated.json for RMA consumption
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

from app.server.models.schemas import ExpertMetadata, YellowPage

logger = logging.getLogger(__name__)


def extract_metadata_from_principles(principles_path: Path) -> Optional[ExpertMetadata]:
    """Extract structured metadata from principles.md file.

    Args:
        principles_path: Path to the principles.md file

    Returns:
        ExpertMetadata if extraction successful, None otherwise
    """
    try:
        content = principles_path.read_text(encoding='utf-8')

        # Look for metadata JSON block in the file
        # Expected format:
        # ## Metadata
        # ```json
        # {
        #   "expert_id": "...",
        #   "description": "...",
        #   ...
        # }
        # ```

        # Pattern to match JSON code block after "## Metadata" or "# Metadata"
        pattern = r'##?\s*Metadata\s*```json\s*(\{[^`]+\})\s*```'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if not match:
            logger.warning(f"No metadata JSON block found in {principles_path}")
            return None

        metadata_json = match.group(1)
        metadata_dict = json.loads(metadata_json)

        # Validate required fields
        required_fields = ["expert_id", "description"]
        for field in required_fields:
            if field not in metadata_dict:
                logger.error(f"Missing required field '{field}' in {principles_path}")
                return None

        # Create ExpertMetadata instance
        expert_metadata = ExpertMetadata(
            expert_id=metadata_dict["expert_id"],
            description=metadata_dict["description"],
            tags=metadata_dict.get("tags", []),
            skill_handles=metadata_dict.get("skill_handles", []),
            output_modes=metadata_dict.get("output_modes", [])
        )

        logger.info(f"Extracted metadata for expert: {expert_metadata.expert_id}")
        return expert_metadata

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in {principles_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error extracting metadata from {principles_path}: {e}")
        return None


def scan_experts_directory(experts_root: Path) -> List[ExpertMetadata]:
    """Scan experts/ directory and extract metadata from all experts.

    Args:
        experts_root: Path to the experts/ directory

    Returns:
        List of ExpertMetadata for all valid experts
    """
    experts = []

    if not experts_root.exists():
        logger.warning(f"Experts directory not found: {experts_root}")
        return experts

    # Iterate through all subdirectories in experts/
    for expert_dir in experts_root.iterdir():
        if not expert_dir.is_dir():
            continue

        # Skip hidden directories and special directories
        if expert_dir.name.startswith('.') or expert_dir.name.startswith('_'):
            logger.debug(f"Skipping special directory: {expert_dir.name}")
            continue

        # Look for principles.md
        principles_path = expert_dir / "principles.md"
        if not principles_path.exists():
            logger.warning(f"No principles.md found in {expert_dir}")
            continue

        # Extract metadata
        metadata = extract_metadata_from_principles(principles_path)
        if metadata:
            experts.append(metadata)

    logger.info(f"Found {len(experts)} valid experts")
    return experts


def generate_yellow_page(experts_root: Path, output_path: Optional[Path] = None) -> YellowPage:
    """Generate yellow page registry from experts directory.

    Args:
        experts_root: Path to the experts/ directory
        output_path: Optional path to write yellow_page.generated.json
                    If None, defaults to .metadata/experts/yellow_page.generated.json

    Returns:
        YellowPage instance
    """
    # Scan experts
    experts = scan_experts_directory(experts_root)

    # Create YellowPage
    yellow_page = YellowPage(experts=experts)

    # Determine output path
    if output_path is None:
        # v3.0.1: Use .metadata/experts/ instead of experts/yellow_pages/
        metadata_dir = Path(".metadata/experts")
        metadata_dir.mkdir(parents=True, exist_ok=True)
        output_path = metadata_dir / "yellow_page.generated.json"

    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(yellow_page.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"Yellow page generated successfully: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write yellow page to {output_path}: {e}")

    return yellow_page


def load_yellow_page(yellow_page_path: Path) -> Optional[YellowPage]:
    """Load yellow page from file.

    Args:
        yellow_page_path: Path to yellow_page.generated.json

    Returns:
        YellowPage instance if successful, None otherwise
    """
    try:
        with open(yellow_page_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        yellow_page = YellowPage(**data)
        logger.info(f"Loaded yellow page with {len(yellow_page.experts)} experts")
        return yellow_page
    except FileNotFoundError:
        logger.error(f"Yellow page file not found: {yellow_page_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load yellow page from {yellow_page_path}: {e}")
        return None


def get_expert_by_id(yellow_page: YellowPage, expert_id: str) -> Optional[ExpertMetadata]:
    """Get expert metadata by expert_id.

    Args:
        yellow_page: YellowPage instance
        expert_id: Expert identifier

    Returns:
        ExpertMetadata if found, None otherwise
    """
    for expert in yellow_page.experts:
        if expert.expert_id == expert_id:
            return expert
    return None


def filter_experts_by_tags(yellow_page: YellowPage, tags: List[str]) -> List[ExpertMetadata]:
    """Filter experts by tags (any match).

    Args:
        yellow_page: YellowPage instance
        tags: List of tags to match

    Returns:
        List of experts that have at least one matching tag
    """
    matching_experts = []
    for expert in yellow_page.experts:
        if any(tag in expert.tags for tag in tags):
            matching_experts.append(expert)
    return matching_experts


def filter_experts_by_output_mode(yellow_page: YellowPage, output_mode: str) -> List[ExpertMetadata]:
    """Filter experts by output mode.

    Args:
        yellow_page: YellowPage instance
        output_mode: Output mode to match (e.g., "suitability_judgment")

    Returns:
        List of experts that support the specified output mode
    """
    matching_experts = []
    for expert in yellow_page.experts:
        if output_mode in expert.output_modes:
            matching_experts.append(expert)
    return matching_experts


# Startup hook
def initialize_yellow_page(experts_root: Path) -> YellowPage:
    """Initialize yellow page at application startup.

    This function should be called when the server starts.

    Args:
        experts_root: Path to the experts/ directory

    Returns:
        YellowPage instance
    """
    logger.info("Initializing yellow page registry...")
    yellow_page = generate_yellow_page(experts_root)
    logger.info(f"Yellow page initialized with {len(yellow_page.experts)} experts:")
    for expert in yellow_page.experts:
        logger.info(f"  - {expert.expert_id}: {expert.description}")
    return yellow_page
