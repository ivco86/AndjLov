"""
PDF Catalog Generator for AI Gallery
Generates professional PDF catalogs with image thumbnails and metadata
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image as PILImage
import os
from datetime import datetime
from io import BytesIO


class PDFCatalogGenerator:
    """Generate PDF catalogs from image metadata"""

    def __init__(self, page_size=A4, orientation='portrait'):
        """
        Initialize PDF catalog generator

        Args:
            page_size: Page size (A4, letter, etc.)
            orientation: 'portrait' or 'landscape'
        """
        self.page_size = page_size
        if orientation == 'landscape':
            self.page_size = (page_size[1], page_size[0])

        self.width, self.height = self.page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CatalogTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CatalogSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#7f8c8d'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))

        # Image description style
        self.styles.add(ParagraphStyle(
            name='ImageDescription',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=6,
            fontName='Helvetica'
        ))

        # Metadata style
        self.styles.add(ParagraphStyle(
            name='Metadata',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#95a5a6'),
            spaceAfter=4,
            fontName='Helvetica'
        ))

    def _get_image_path(self, image_data, data_dir):
        """Get the full image path"""
        filepath = image_data.get('filepath', '')
        if os.path.isabs(filepath):
            return filepath
        return os.path.join(data_dir, filepath)

    def _add_image_to_story(self, story, image_data, data_dir, thumbnail_dir, max_width=4*inch):
        """Add an image with metadata to the story"""
        # Try to get image path
        image_path = self._get_image_path(image_data, data_dir)

        # Check for thumbnail first
        thumbnail_path = None
        if thumbnail_dir and image_data.get('id'):
            thumbnail_filename = f"{image_data['id']}.jpg"
            potential_thumbnail = os.path.join(thumbnail_dir, thumbnail_filename)
            if os.path.exists(potential_thumbnail):
                thumbnail_path = potential_thumbnail

        # Use thumbnail if available, otherwise original image
        img_path_to_use = thumbnail_path if thumbnail_path else image_path

        # Add image if file exists
        if os.path.exists(img_path_to_use):
            try:
                # Open image to get dimensions
                with PILImage.open(img_path_to_use) as pil_img:
                    img_width, img_height = pil_img.size

                    # Calculate scaled dimensions
                    aspect_ratio = img_height / img_width
                    display_width = min(max_width, img_width)
                    display_height = display_width * aspect_ratio

                    # Add image
                    img = Image(img_path_to_use, width=display_width, height=display_height)
                    story.append(img)
                    story.append(Spacer(1, 0.1*inch))
            except Exception as e:
                # If image can't be loaded, add placeholder text
                story.append(Paragraph(f"<i>[Image: {image_data.get('filename', 'Unknown')}]</i>",
                                     self.styles['ImageDescription']))

        # Add filename
        filename = image_data.get('filename', 'Unknown')
        story.append(Paragraph(f"<b>{filename}</b>", self.styles['ImageDescription']))

        # Add description if available
        description = image_data.get('description', '')
        if description:
            story.append(Paragraph(description, self.styles['ImageDescription']))

        # Add metadata
        metadata_parts = []

        # Dimensions
        width = image_data.get('width')
        height = image_data.get('height')
        if width and height:
            metadata_parts.append(f"Size: {width}x{height}px")

        # File size
        file_size = image_data.get('file_size')
        if file_size:
            size_mb = file_size / (1024 * 1024)
            if size_mb >= 1:
                metadata_parts.append(f"File: {size_mb:.2f} MB")
            else:
                size_kb = file_size / 1024
                metadata_parts.append(f"File: {size_kb:.2f} KB")

        # Tags
        tags = image_data.get('tags', '')
        if tags:
            metadata_parts.append(f"Tags: {tags}")

        # Created date
        created_at = image_data.get('created_at', '')
        if created_at:
            metadata_parts.append(f"Added: {created_at[:10]}")

        if metadata_parts:
            metadata_text = " | ".join(metadata_parts)
            story.append(Paragraph(metadata_text, self.styles['Metadata']))

        # Add separator
        story.append(HRFlowable(width="80%", thickness=0.5, color=colors.HexColor('#ecf0f1'),
                               spaceAfter=0.2*inch, spaceBefore=0.2*inch))

    def generate_catalog(self, images, output_path, title="Image Catalog",
                        subtitle=None, data_dir=".", thumbnail_dir=None):
        """
        Generate a PDF catalog from image metadata

        Args:
            images: List of image dictionaries with metadata
            output_path: Path where PDF should be saved
            title: Catalog title
            subtitle: Optional subtitle
            data_dir: Base directory for image files
            thumbnail_dir: Directory containing thumbnails

        Returns:
            str: Path to generated PDF
        """
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Build story
        story = []

        # Add title
        story.append(Paragraph(title, self.styles['CatalogTitle']))

        # Add subtitle if provided
        if subtitle:
            story.append(Paragraph(subtitle, self.styles['CatalogSubtitle']))
        else:
            # Default subtitle with date and image count
            default_subtitle = f"Generated on {datetime.now().strftime('%B %d, %Y')} | {len(images)} images"
            story.append(Paragraph(default_subtitle, self.styles['CatalogSubtitle']))

        story.append(Spacer(1, 0.3*inch))

        # Add each image
        for idx, image_data in enumerate(images):
            self._add_image_to_story(story, image_data, data_dir, thumbnail_dir)

            # Add page break every 3 images (except for the last one)
            if (idx + 1) % 3 == 0 and idx < len(images) - 1:
                story.append(PageBreak())

        # Build PDF
        doc.build(story)

        return output_path

    def generate_board_catalog(self, board_info, images, output_path,
                              data_dir=".", thumbnail_dir=None):
        """
        Generate a PDF catalog for a specific board

        Args:
            board_info: Dictionary with board metadata (name, description)
            images: List of image dictionaries
            output_path: Path where PDF should be saved
            data_dir: Base directory for image files
            thumbnail_dir: Directory containing thumbnails

        Returns:
            str: Path to generated PDF
        """
        title = f"{board_info.get('name', 'Board')} Catalog"
        subtitle = board_info.get('description', '')

        if not subtitle:
            subtitle = f"Generated on {datetime.now().strftime('%B %d, %Y')} | {len(images)} images"
        else:
            subtitle += f" | {len(images)} images | {datetime.now().strftime('%B %d, %Y')}"

        return self.generate_catalog(
            images=images,
            output_path=output_path,
            title=title,
            subtitle=subtitle,
            data_dir=data_dir,
            thumbnail_dir=thumbnail_dir
        )
