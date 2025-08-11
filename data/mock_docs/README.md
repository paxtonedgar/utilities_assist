# Mock Documents for Local Development

This directory contains 50 mock JSON documents that mimic the Confluence scrape structure expected by the production OpenSearch index.

## Document Structure

Each document follows this schema:

```json
{
  "_id": "doc-001",
  "page_url": "https://docs.example.com/service/topic",
  "api_name": "Service Name - API Type",
  "utility_name": "Service Category",
  "sections": [
    {
      "heading": "Section Title",
      "content": "Section content with detailed information..."
    }
  ],
  "created_date": "2024-01-01",
  "last_updated": "2024-01-15", 
  "doc_type": "api_documentation",
  "tags": ["category", "service", "documentation"],
  "version": "1.0"
}
```

## Fields Explained

### Required for Search System
- **`_id`**: Unique document identifier
- **`page_url`**: Source URL (used in search results)
- **`api_name`**: API/service name (used for filtering)
- **`utility_name`**: Utility category (used for filtering)
- **`sections`**: Array of content sections with:
  - `heading`: Section title
  - `content`: Main text content
  - `embedding`: Vector embedding (added during indexing)

### Additional Metadata
- **`created_date`**: Document creation date
- **`last_updated`**: Last modification date
- **`doc_type`**: Document type classification
- **`tags`**: Array of categorization tags
- **`version`**: Document version

## Content Sources

Documents contain realistic content from public documentation sources:

- **Streamlit Documentation**: UI components, caching, deployment
- **Kubernetes Documentation**: Pods, services, deployments, ingress
- **API Best Practices**: Authentication, rate limiting, versioning, error handling
- **Database Management**: Indexing, transactions, backup/recovery
- **Security Guidelines**: OAuth 2.0, encryption, input validation

## Files

- **`doc-001.json` to `doc-050.json`**: Individual document files
- **`all_documents.json`**: Combined file with all 50 documents
- **`index_template.json`**: OpenSearch index template/mapping
- **`README.md`**: This documentation file

## Usage

### Local Development
```bash
# Use mock search with these documents
UTILITIES_CONFIG=config.local.ini USE_MOCK_SEARCH=true make run-local
```

### Indexing to OpenSearch (if needed)
```bash
# The index template shows the expected mapping:
curl -X PUT "localhost:9200/_index_template/utilities-template" \
  -H "Content-Type: application/json" \
  -d @index_template.json
```

### Validation
```bash
# Validate all documents against schema
python validate_mock_docs.py
```

## Document Categories

The mock documents cover these utility categories:
- Authentication Service
- Data Pipeline  
- Search Engine
- User Management
- Configuration Service
- Monitoring Dashboard
- API Gateway
- Content Management
- Notification Service
- File Storage
- Analytics Platform
- Deployment Pipeline

Each document contains 1-4 sections with varied content to provide realistic search testing scenarios.