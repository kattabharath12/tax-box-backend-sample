import re
import os
from datetime import datetime
from typing import Dict, Any, List

class DocumentProcessor:
    def __init__(self):
        # Simple patterns for basic text extraction
        self.common_patterns = {
            'amount': r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            'ssn': r'\d{3}-\d{2}-\d{4}',
            'year': r'20\d{2}',
            'number': r'\d+(?:,\d{3})*(?:\.\d{2})?'
        }
        
        # Document type keywords
        self.document_keywords = {
            'w2': ['w2', 'w-2', 'wage', 'salary', 'withholding'],
            '1099': ['1099', '1099-misc', '1099-nec', 'nonemployee', 'contractor'],
            '1098': ['1098', 'mortgage', 'interest', 'loan'],
            'receipt': ['receipt', 'expense', 'deduction'],
            'bank': ['bank', 'statement', 'account']
        }

    def process_document(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Process document with simple classification"""
        try:
            # Get document info
            filename = os.path.basename(file_path).lower()
            
            # Classify document type
            doc_type = self._classify_document(filename)
            
            # Get file size for basic validation
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Generate suggestions based on document type
            suggestions = self._get_document_suggestions(doc_type, filename)
            
            # Basic extracted data
            extracted_data = {
                'filename': filename,
                'file_type': file_type,
                'file_size': file_size,
                'classification_confidence': self._get_confidence(doc_type, filename)
            }
            
            return {
                "success": True,
                "document_type": doc_type,
                "extracted_data": extracted_data,
                "suggestions": suggestions,
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}

    def _classify_document(self, filename: str) -> str:
        """Classify document based on filename"""
        filename_lower = filename.lower()
        
        # Check each document type
        for doc_type, keywords in self.document_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                return doc_type
        
        return 'unknown'

    def _get_confidence(self, doc_type: str, filename: str) -> float:
        """Get classification confidence"""
        if doc_type == 'unknown':
            return 0.3
        
        # Count matching keywords
        keywords = self.document_keywords.get(doc_type, [])
        matches = sum(1 for keyword in keywords if keyword in filename.lower())
        
        # Higher confidence for more matches
        return min(0.9, 0.5 + (matches * 0.2))

    def _get_document_suggestions(self, doc_type: str, filename: str) -> List[Dict]:
        """Generate helpful suggestions based on document type"""
        suggestions = []
        
        if doc_type == 'w2':
            suggestions = [
                {
                    'field': 'income',
                    'description': 'Enter wages from W-2 (Box 1)',
                    'confidence': 0.8,
                    'suggested_value': None,
                    'help_text': 'Look for "Wages, tips, other compensation" on your W-2'
                },
                {
                    'field': 'withholdings',
                    'description': 'Enter federal tax withheld (Box 2)',
                    'confidence': 0.8,
                    'suggested_value': None,
                    'help_text': 'Look for "Federal income tax withheld" on your W-2'
                }
            ]
        elif doc_type == '1099':
            suggestions = [
                {
                    'field': 'income',
                    'description': 'Enter 1099 income',
                    'confidence': 0.7,
                    'suggested_value': None,
                    'help_text': 'Look for "Nonemployee compensation" or similar on your 1099'
                }
            ]
        elif doc_type == '1098':
            suggestions = [
                {
                    'field': 'deductions',
                    'description': 'Enter mortgage interest paid',
                    'confidence': 0.7,
                    'suggested_value': None,
                    'help_text': 'This can be deducted if you itemize deductions'
                }
            ]
        elif doc_type == 'receipt':
            suggestions = [
                {
                    'field': 'deductions',
                    'description': 'Consider if this is a deductible expense',
                    'confidence': 0.5,
                    'suggested_value': None,
                    'help_text': 'Business expenses may be deductible'
                }
            ]
        else:
            suggestions = [
                {
                    'field': 'general',
                    'description': 'Document uploaded successfully',
                    'confidence': 0.5,
                    'suggested_value': None,
                    'help_text': 'Please review this document and enter relevant information manually'
                }
            ]
        
        return suggestions

    def suggest_tax_entries(self, extracted_data: Dict) -> List[Dict]:
        """Convert extracted data to tax entry suggestions"""
        return extracted_data.get('suggestions', [])

    def get_document_analysis(self, file_path: str) -> Dict[str, Any]:
        """Get additional document analysis"""
        try:
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            return {
                'readable': True,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'recommendations': self._get_recommendations(filename),
                'next_steps': self._get_next_steps(filename)
            }
        except Exception as e:
            return {'error': str(e)}

    def _get_recommendations(self, filename: str) -> List[str]:
        """Get recommendations based on document type"""
        doc_type = self._classify_document(filename)
        
        recommendations = {
            'w2': [
                "Ensure you enter all W-2 forms if you have multiple employers",
                "Check if you qualify for any tax credits",
                "Consider contributing to retirement accounts"
            ],
            '1099': [
                "Keep track of business expenses if this is contractor income",
                "Consider quarterly tax payments for next year",
                "You may need to pay self-employment tax"
            ],
            '1098': [
                "Consider if itemizing deductions would be beneficial",
                "Keep records of property taxes paid",
                "PMI may also be deductible"
            ],
            'unknown': [
                "Review the document to identify tax-relevant information",
                "Consider organizing documents by tax category",
                "Keep all tax documents until you file"
            ]
        }
        
        return recommendations.get(doc_type, recommendations['unknown'])

    def _get_next_steps(self, filename: str) -> List[str]:
        """Get next steps for the user"""
        return [
            "Review the suggested tax entries below",
            "Enter the amounts in your tax return",
            "Keep the original document for your records",
            "Consider consulting a tax professional for complex situations"
        ]