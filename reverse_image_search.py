"""
Reverse Image Search Module
Supports multiple search engines for finding image sources
"""

import os
import base64
import requests
from typing import List, Dict, Optional
from urllib.parse import urlencode


class ReverseImageSearch:
    """Reverse image search using multiple providers"""

    @staticmethod
    def search_google(image_path: str) -> Dict:
        """
        Search using Google Images (via URL construction)
        Note: This creates a search URL rather than using an API
        """
        # Google Images search by image URL
        # For local images, we would need to upload them or use base64
        # This returns a search URL that can be opened

        return {
            'provider': 'Google Images',
            'method': 'url',
            'url': None,  # Will be constructed when image is accessible
            'instructions': 'Upload image to Google Images to search',
            'search_url': 'https://images.google.com/searchbyimage'
        }

    @staticmethod
    def search_tineye(image_path: str, api_key: Optional[str] = None) -> Dict:
        """
        Search using TinEye API
        Requires API key for programmatic access
        """
        if not api_key:
            return {
                'provider': 'TinEye',
                'method': 'url',
                'url': 'https://tineye.com/',
                'instructions': 'Visit TinEye.com and upload the image manually',
                'requires_api_key': True
            }

        # With API key, could make actual API calls
        # https://api.tineye.com/rest/

        return {
            'provider': 'TinEye',
            'method': 'api',
            'status': 'not_implemented',
            'message': 'TinEye API integration requires API key configuration'
        }

    @staticmethod
    def search_yandex(image_path: str) -> Dict:
        """
        Search using Yandex Images
        Returns URL for manual search
        """
        return {
            'provider': 'Yandex Images',
            'method': 'url',
            'url': 'https://yandex.com/images/search',
            'instructions': 'Visit Yandex Images and use the camera icon to upload',
            'search_url': 'https://yandex.com/images/search?rpt=imageview'
        }

    @staticmethod
    def search_bing(image_path: str) -> Dict:
        """
        Search using Bing Visual Search
        Returns URL for manual search
        """
        return {
            'provider': 'Bing Visual Search',
            'method': 'url',
            'url': 'https://www.bing.com/visualsearch',
            'instructions': 'Visit Bing Visual Search and upload the image',
            'search_url': 'https://www.bing.com/visualsearch'
        }

    @staticmethod
    def get_image_base64(image_path: str) -> Optional[str]:
        """Convert image to base64 for embedding"""
        try:
            with open(image_path, 'rb') as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                return encoded
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    @staticmethod
    def get_all_search_options(image_id: int, image_url: str) -> List[Dict]:
        """
        Get all available reverse image search options

        Args:
            image_id: ID of the image in the database
            image_url: URL to access the image (e.g., /api/images/123/file)

        Returns:
            List of search options with URLs and instructions
        """
        # Construct URLs that will work with the local server
        # These are designed to be opened in new tabs

        options = [
            {
                'provider': 'Google Images',
                'icon': '🔍',
                'description': 'Most comprehensive search engine',
                'instructions': 'Click to open Google Images. Then drag and drop the image or click the camera icon to upload.',
                'url': 'https://images.google.com/',
                'search_url': 'https://lens.google.com/uploadbyurl',
                'direct_search': False,
                'features': ['Find similar images', 'Discover sources', 'Copyright info', 'Related searches']
            },
            {
                'provider': 'TinEye',
                'icon': '👁️',
                'description': 'Specialized reverse image search',
                'instructions': 'Click to open TinEye. Upload your image to find where it appears online.',
                'url': 'https://tineye.com/',
                'search_url': None,
                'direct_search': False,
                'features': ['Track image usage', 'Find manipulations', 'Oldest version', 'Most changed']
            },
            {
                'provider': 'Yandex Images',
                'icon': '🔎',
                'description': 'Good for Russian and European sources',
                'instructions': 'Click to open Yandex Images. Click the camera icon and upload your image.',
                'url': 'https://yandex.com/images/',
                'search_url': None,
                'direct_search': False,
                'features': ['Russian sources', 'European content', 'Face search', 'Similar images']
            },
            {
                'provider': 'Bing Visual Search',
                'icon': '🌐',
                'description': 'Microsoft\'s visual search engine',
                'instructions': 'Click to open Bing Visual Search. Upload your image to find similar images and sources.',
                'url': 'https://www.bing.com/visualsearch',
                'search_url': None,
                'direct_search': False,
                'features': ['Visual matches', 'Related products', 'Image info', 'Web sources']
            },
            {
                'provider': 'Image Download',
                'icon': '💾',
                'description': 'Download image for manual search',
                'instructions': 'Download the image to your computer, then upload it to any search engine.',
                'url': image_url,
                'download': True,
                'direct_search': False,
                'features': ['Use with any search engine', 'Keep a copy', 'Edit before searching']
            }
        ]

        return options

    @staticmethod
    def create_search_guide() -> Dict:
        """
        Create a guide for how to use reverse image search effectively
        """
        return {
            'title': 'Reverse Image Search Guide',
            'sections': [
                {
                    'title': 'What is Reverse Image Search?',
                    'content': 'Upload an image to find where it appears online, discover its source, and check for copyright information.'
                },
                {
                    'title': 'Best Practices',
                    'tips': [
                        'Use multiple search engines for comprehensive results',
                        'Try Google Images first for the widest coverage',
                        'Use TinEye to track how images are used',
                        'Check Yandex for Russian and European sources',
                        'Compare results across different engines'
                    ]
                },
                {
                    'title': 'Copyright Checking',
                    'tips': [
                        'Look for original source and publication date',
                        'Check image metadata and EXIF data',
                        'Search for photographer or creator credits',
                        'Look for Creative Commons or licensing info',
                        'When in doubt, contact the image owner'
                    ]
                },
                {
                    'title': 'Finding Original Source',
                    'tips': [
                        'Look for oldest appearance of the image',
                        'Check high-resolution versions',
                        'Look for watermarks or signatures',
                        'Check professional photography sites',
                        'Use image date filters when available'
                    ]
                }
            ]
        }


def get_copyright_tips() -> List[str]:
    """Get tips for checking image copyright"""
    return [
        "🔍 Search for the original source and publication date",
        "👤 Look for photographer or creator credits",
        "📜 Check for Creative Commons or licensing information",
        "⏰ Find the oldest appearance of the image online",
        "📧 Contact the copyright holder if you want to use the image",
        "🚫 Assume all images are copyrighted unless stated otherwise",
        "✅ Use royalty-free or public domain images when possible"
    ]


def get_usage_detection_tips() -> List[str]:
    """Get tips for detecting unauthorized usage"""
    return [
        "🔄 Regularly search for your images online",
        "📸 Use watermarks on published photos",
        "📝 Keep records of original image metadata",
        "⚖️ Know your rights as an image creator",
        "🛡️ Consider using DMCA takedown for violations",
        "💼 Set up Google Alerts for your image titles",
        "🔐 Register copyrights for valuable images"
    ]
