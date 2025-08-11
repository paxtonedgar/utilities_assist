# Blue/Green Reindex for Confluence v2

## Overview

This blue/green reindexing system allows zero-downtime migrations from existing Confluence indices to new v2 mappings with root-level vector embeddings.

## Files

- `src/search/mappings/confluence_v2.json` - New mapping with 1536-dim root vectors
- `scripts/reindex_blue_green.py` - Reindex script with blue/green deployment
- `scripts/test_reindex.py` - Validation tests

## Key Features

### Mapping v2 (`confluence_v2.json`)
- **Root-level embedding**: 1536-dimensional dense_vector with HNSW indexing
- **Optimized fields**: title (text), section (keyword), body (text), updated_at (date)
- **Metadata support**: page_id, section_anchor, canonical_id, acl_hash
- **Performance tuned**: 2 shards, 1 replica, 15s refresh interval
- **English analyzer**: Stopwords and text optimization

### Reindex Script (`reindex_blue_green.py`)
- **Zero-downtime**: Creates new index, migrates data, updates alias atomically
- **Document transformation**: Handles nested→root embedding migration
- **Dry-run mode**: Test without making changes
- **Comprehensive logging**: Progress tracking and error reporting
- **Cleanup**: Automatic removal of old indices (keeps 3 most recent)

## Usage

### Basic Reindex
```bash
# Dry run first (recommended)
python scripts/reindex_blue_green.py --dry-run

# Execute reindex
python scripts/reindex_blue_green.py
```

### Advanced Options
```bash
# Custom source index and batch size
python scripts/reindex_blue_green.py \
    --source-index my-old-index \
    --batch-size 500 \
    --log-level DEBUG

# Dry run with different settings
python scripts/reindex_blue_green.py \
    --dry-run \
    --source-index khub-test-md \
    --batch-size 200
```

## Process Flow

1. **Validation**: Check source index exists and get statistics
2. **Creation**: Create new `confluence_v2-TIMESTAMP` index with v2 mappings  
3. **Migration**: Transform and reindex all documents in batches
4. **Alias Update**: Atomically update `confluence_current` alias
5. **Cleanup**: Remove old indices (keeps 3 most recent)

## Document Transformation

The script automatically handles:

```python
# Old nested structure
{
    "title": "Page Title",
    "content": "Body text", 
    "nested_content": [
        {"embedding": [0.1, 0.2, ...]}  # 1536 dims
    ]
}

# New root structure  
{
    "title": "Page Title",
    "body": "Body text",
    "embedding": [0.1, 0.2, ...],  # Root level
    "section": "main",
    "updated_at": "2024-01-01T00:00:00Z",
    "page_id": "page123",
    "canonical_id": "doc456",
    "acl_hash": "public"
}
```

## Monitoring

The script provides comprehensive output:

```
🚀 Starting blue/green reindex
Source: khub-test-md
Target: confluence_v2-20240111_143022
📊 Source index stats: 1,247 docs, 45.2 MB
✅ Target index created successfully
Indexed 1000 documents...
✅ Alias updated successfully
✅ Blue/green reindex completed successfully
📈 Documents processed: 1,247
⏱️  Duration: 23.45 seconds
```

## Error Handling

- **Connection failures**: Automatic retries with exponential backoff
- **Document errors**: Individual failures logged but don't stop process
- **Validation**: Pre-flight checks for source index existence
- **Rollback**: On failure, old alias remains unchanged

## Performance Tuning

### Batch Processing
- Default: 100 docs/batch
- Recommended: 200-500 for large datasets
- Memory limit: 50MB chunks

### Parallel Processing  
- 4 worker threads by default
- Configurable through `parallel_bulk`
- Balances speed vs cluster load

### Index Settings
- **Shards**: 2 (good for medium datasets)
- **Replicas**: 1 (balance availability/storage) 
- **Refresh**: 15s (reduces indexing overhead)

## Testing

```bash
# Run validation tests
python scripts/test_reindex.py

# Test with small dataset
python scripts/reindex_blue_green.py --dry-run --batch-size 10

# Monitor cluster during reindex
curl -X GET "localhost:9200/_cluster/health?pretty"
curl -X GET "localhost:9200/_cat/indices/confluence_v2-*?v"
```

## Troubleshooting

### Common Issues

1. **"Source index not found"**
   ```bash
   # Check available indices
   curl -X GET "localhost:9200/_cat/indices?v"
   # Specify correct source
   python scripts/reindex_blue_green.py --source-index correct-name
   ```

2. **"Connection refused"** 
   - Check OpenSearch is running
   - Verify settings in `.env` or config files
   - Test connection: `curl localhost:9200`

3. **Memory errors during large reindex**
   ```bash
   # Reduce batch size
   python scripts/reindex_blue_green.py --batch-size 50
   ```

4. **Mapping conflicts**
   - Check source data format matches expected v2 schema
   - Review transformation logic in `transform_document()`

### Recovery

If reindex fails partway through:
1. Check error logs for specific issues
2. Fix underlying problems  
3. Delete failed target index: `curl -X DELETE localhost:9200/confluence_v2-TIMESTAMP`
4. Re-run reindex

## Security

- **ACL Support**: Maintains access control hashes
- **Authentication**: Supports OpenSearch security plugin
- **Validation**: Input sanitization and type checking
- **Audit Trail**: Complete logging of operations

## Integration

The reindex script integrates with the existing application:

```python
# Use new alias in search services
from src.services.retrieve import search_documents

# Automatically uses confluence_current alias
results = await search_documents("user query", index="confluence_current")
```

After successful reindex, all application searches automatically use the new v2 index structure.