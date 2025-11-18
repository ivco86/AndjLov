"""
Research & Education Service
Provides citation generation and dataset export for ML training
"""

import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import io


class ResearchService:
    """Service for research and education features"""

    def __init__(self, database):
        self.db = database

    # ============ CITATION GENERATION ============

    def generate_citation(self, image: Dict, format_type: str = 'apa') -> str:
        """
        Generate citation for an image in various formats

        Args:
            image: Image dictionary from database
            format_type: 'apa', 'mla', 'bibtex', 'chicago'

        Returns:
            Formatted citation string
        """
        if format_type == 'apa':
            return self._generate_apa_citation(image)
        elif format_type == 'mla':
            return self._generate_mla_citation(image)
        elif format_type == 'bibtex':
            return self._generate_bibtex_citation(image)
        elif format_type == 'chicago':
            return self._generate_chicago_citation(image)
        else:
            raise ValueError(f"Unsupported citation format: {format_type}")

    def _generate_apa_citation(self, image: Dict) -> str:
        """Generate APA format citation"""
        # Extract metadata
        filename = image.get('filename', 'Untitled')
        description = image.get('description', 'No description')
        created_at = image.get('created_at', datetime.now().isoformat())

        # Parse date
        try:
            date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            year = date_obj.year
        except:
            year = datetime.now().year

        # APA format: Author. (Year). Title [Photograph/Image]. Description.
        title = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()

        return f"AI Gallery. ({year}). {title} [Digital Image]. {description}"

    def _generate_mla_citation(self, image: Dict) -> str:
        """Generate MLA format citation"""
        filename = image.get('filename', 'Untitled')
        description = image.get('description', 'No description')
        created_at = image.get('created_at', datetime.now().isoformat())

        try:
            date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            date_str = date_obj.strftime('%d %b. %Y')
        except:
            date_str = datetime.now().strftime('%d %b. %Y')

        # MLA format: "Title." Description, Date.
        title = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()

        return f'"{title}." {description}, {date_str}.'

    def _generate_bibtex_citation(self, image: Dict) -> str:
        """Generate BibTeX format citation"""
        filename = image.get('filename', 'Untitled')
        description = image.get('description', 'No description')
        created_at = image.get('created_at', datetime.now().isoformat())
        image_id = image.get('id', 0)

        try:
            date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            year = date_obj.year
        except:
            year = datetime.now().year

        # BibTeX key: sanitize filename
        key = Path(filename).stem.replace(' ', '').replace('_', '').replace('-', '')[:20]
        title = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()

        return f"""@misc{{{key}{image_id},
  title = {{{title}}},
  author = {{AI Gallery}},
  year = {{{year}}},
  note = {{{description}}},
  howpublished = {{Digital Image}}
}}"""

    def _generate_chicago_citation(self, image: Dict) -> str:
        """Generate Chicago format citation"""
        filename = image.get('filename', 'Untitled')
        description = image.get('description', 'No description')
        created_at = image.get('created_at', datetime.now().isoformat())

        try:
            date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            year = date_obj.year
        except:
            year = datetime.now().year

        # Chicago format: Author, "Title," Description (Year).
        title = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()

        return f'AI Gallery, "{title}," {description} ({year}).'

    def generate_batch_citations(self, image_ids: List[int], format_type: str = 'apa') -> List[Dict]:
        """Generate citations for multiple images"""
        citations = []

        for image_id in image_ids:
            image = self.db.get_image(image_id)
            if image:
                citation = self.generate_citation(image, format_type)
                citations.append({
                    'image_id': image_id,
                    'filename': image.get('filename'),
                    'citation': citation
                })

        return citations

    # ============ DATASET EXPORT ============

    def export_dataset_coco(self, image_ids: List[int] = None, include_unannotated: bool = False) -> Dict:
        """
        Export dataset in COCO format for object detection

        Returns:
            COCO format dictionary
        """
        # Get all annotations
        all_annotations = self.db.get_all_annotations()

        # Filter by image_ids if provided
        if image_ids:
            all_annotations = [a for a in all_annotations if a['image_id'] in image_ids]

        # Get unique images with annotations
        annotated_image_ids = list(set(a['image_id'] for a in all_annotations))

        # Get image details
        images_data = []
        for img_id in annotated_image_ids:
            image = self.db.get_image(img_id)
            if image:
                images_data.append({
                    'id': image['id'],
                    'file_name': image['filename'],
                    'width': image['width'] or 0,
                    'height': image['height'] or 0,
                    'date_captured': image.get('created_at', ''),
                })

        # Build category list from unique class names
        class_names = sorted(list(set(a['class_name'] for a in all_annotations)))
        categories = [
            {
                'id': idx + 1,
                'name': class_name,
                'supercategory': 'object'
            }
            for idx, class_name in enumerate(class_names)
        ]

        # Map class names to IDs
        class_name_to_id = {cat['name']: cat['id'] for cat in categories}

        # Build annotations list
        annotations_data = []
        for idx, ann in enumerate(all_annotations):
            # COCO uses absolute coordinates
            annotations_data.append({
                'id': idx + 1,
                'image_id': ann['image_id'],
                'category_id': class_name_to_id[ann['class_name']],
                'bbox': [ann['x'], ann['y'], ann['width'], ann['height']],  # x, y, width, height
                'area': ann['width'] * ann['height'],
                'iscrowd': 0,
                'segmentation': []
            })

        # Build COCO format
        coco_data = {
            'info': {
                'description': 'AI Gallery Dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'date_created': datetime.now().isoformat()
            },
            'licenses': [],
            'images': images_data,
            'annotations': annotations_data,
            'categories': categories
        }

        return coco_data

    def export_dataset_yolo(self, image_ids: List[int] = None, output_dir: str = None) -> Dict:
        """
        Export dataset in YOLO format

        Returns:
            Dictionary with YOLO files content
        """
        # Get all annotations
        all_annotations = self.db.get_all_annotations()

        # Filter by image_ids if provided
        if image_ids:
            all_annotations = [a for a in all_annotations if a['image_id'] in image_ids]

        # Get unique class names
        class_names = sorted(list(set(a['class_name'] for a in all_annotations)))
        class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

        # Group annotations by image
        annotations_by_image = {}
        for ann in all_annotations:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        # Generate YOLO format for each image
        yolo_files = {}
        image_list = []

        for img_id, annotations in annotations_by_image.items():
            image = self.db.get_image(img_id)
            if not image:
                continue

            img_width = image.get('width', 1)
            img_height = image.get('height', 1)

            # YOLO format: class_id center_x center_y width height (normalized 0-1)
            yolo_lines = []
            for ann in annotations:
                class_id = class_name_to_id[ann['class_name']]

                # Convert from x, y, w, h to normalized center coordinates
                x_center = (ann['x'] + ann['width'] / 2) / img_width
                y_center = (ann['y'] + ann['height'] / 2) / img_height
                norm_width = ann['width'] / img_width
                norm_height = ann['height'] / img_height

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

            # Store annotation file content
            label_filename = Path(image['filename']).stem + '.txt'
            yolo_files[label_filename] = '\n'.join(yolo_lines)
            image_list.append(image['filepath'])

        # Classes file
        yolo_files['classes.txt'] = '\n'.join(class_names)

        # Data YAML for YOLO training
        yolo_files['data.yaml'] = f"""train: train/images
val: val/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""

        return {
            'files': yolo_files,
            'images': image_list,
            'classes': class_names
        }

    def export_dataset_csv(self, image_ids: List[int] = None) -> str:
        """
        Export dataset in CSV format

        Returns:
            CSV content as string
        """
        import csv
        import io

        # Get all annotations
        all_annotations = self.db.get_all_annotations()

        # Filter by image_ids if provided
        if image_ids:
            all_annotations = [a for a in all_annotations if a['image_id'] in image_ids]

        # Generate CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'image_id', 'filename', 'filepath', 'img_width', 'img_height',
            'annotation_id', 'class_name', 'x', 'y', 'width', 'height', 'confidence'
        ])

        # Data rows
        for ann in all_annotations:
            writer.writerow([
                ann['image_id'],
                ann['filename'],
                ann['filepath'],
                ann['img_width'],
                ann['img_height'],
                ann['id'],
                ann['class_name'],
                ann['x'],
                ann['y'],
                ann['width'],
                ann['height'],
                ann.get('confidence', 1.0)
            ])

        return output.getvalue()

    def create_dataset_splits(self, image_ids: List[int], train_ratio: float = 0.7,
                             val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, List[int]]:
        """
        Split dataset into train/val/test sets

        Returns:
            Dictionary with 'train', 'val', 'test' lists of image IDs
        """
        import random

        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")

        # Shuffle image IDs
        shuffled_ids = image_ids.copy()
        random.shuffle(shuffled_ids)

        # Calculate split indices
        total = len(shuffled_ids)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        return {
            'train': shuffled_ids[:train_end],
            'val': shuffled_ids[train_end:val_end],
            'test': shuffled_ids[val_end:]
        }

    def export_dataset_zip(self, format_type: str, image_ids: List[int] = None,
                          include_images: bool = False, split_dataset: bool = False) -> bytes:
        """
        Export complete dataset as ZIP file

        Args:
            format_type: 'coco', 'yolo', 'csv'
            image_ids: List of image IDs to export (None = all)
            include_images: Include actual image files in ZIP
            split_dataset: Create train/val/test splits

        Returns:
            ZIP file as bytes
        """
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            if format_type == 'coco':
                coco_data = self.export_dataset_coco(image_ids)
                zip_file.writestr('annotations.json', json.dumps(coco_data, indent=2))

                if split_dataset and image_ids:
                    splits = self.create_dataset_splits(image_ids)
                    for split_name, split_ids in splits.items():
                        split_data = self.export_dataset_coco(split_ids)
                        zip_file.writestr(f'{split_name}/annotations.json', json.dumps(split_data, indent=2))

            elif format_type == 'yolo':
                yolo_data = self.export_dataset_yolo(image_ids)

                # Write label files
                for filename, content in yolo_data['files'].items():
                    zip_file.writestr(f'labels/{filename}', content)

            elif format_type == 'csv':
                csv_data = self.export_dataset_csv(image_ids)
                zip_file.writestr('annotations.csv', csv_data)

            # Add README
            readme = self._generate_dataset_readme(format_type)
            zip_file.writestr('README.md', readme)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    def _generate_dataset_readme(self, format_type: str) -> str:
        """Generate README for exported dataset"""
        stats = self.db.get_dataset_statistics()

        readme = f"""# AI Gallery Dataset Export

## Dataset Information

- **Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Format**: {format_type.upper()}
- **Total Images**: {stats['annotated_images']}
- **Total Annotations**: {stats['total_annotations']}

## Class Distribution

"""
        for class_info in stats['class_distribution']:
            readme += f"- **{class_info['class_name']}**: {class_info['count']} annotations\n"

        if format_type == 'coco':
            readme += """
## COCO Format

The COCO format JSON file contains:
- `images`: List of images with metadata
- `annotations`: Bounding box annotations
- `categories`: Object classes

Use with: PyTorch, TensorFlow, Detectron2, MMDetection
"""
        elif format_type == 'yolo':
            readme += """
## YOLO Format

- `labels/*.txt`: Annotation files (one per image)
- `classes.txt`: List of class names
- `data.yaml`: YOLO configuration file

Format: `class_id center_x center_y width height` (normalized 0-1)

Use with: YOLOv5, YOLOv8, Ultralytics
"""
        elif format_type == 'csv':
            readme += """
## CSV Format

Simple CSV file with columns:
- image_id, filename, filepath, img_width, img_height
- annotation_id, class_name, x, y, width, height, confidence

Use with: Pandas, Excel, custom scripts
"""

        readme += """
## Citation

If you use this dataset in your research, please cite:

```
AI Gallery Dataset ({year})
Exported from AI Gallery
Date: {date}
```

## License

Please ensure you have the rights to use these images for your intended purpose.
""".format(year=datetime.now().year, date=datetime.now().strftime('%Y-%m-%d'))

        return readme


# Global instance
_research_service = None


def get_research_service(database=None):
    """Get or create research service singleton"""
    global _research_service

    if _research_service is None and database:
        _research_service = ResearchService(database)

    return _research_service
