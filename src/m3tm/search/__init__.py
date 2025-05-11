"""
M³TM Arama Modülü

Bu modül, M³TM modelinin semantik arama özelliklerini sağlar. Arama gömme projeksiyonu,
indeksleme ve sorgu işleme bileşenlerini içerir.
"""

from .projection import (
    SearchProjectionConfig,
    M3TMSearchProjection,
    SearchProjectionFactory
)

from .index import (
    SearchIndexConfig,
    SearchIndex,
    SearchIndexFactory
)

from .api import (
    SearchServiceConfig,
    SearchService,
    SearchServiceFactory,
    SearchFilter,
    Pagination,
    SearchResult,
    SearchResults,
    SearchResultType
)

__all__ = [
    'SearchProjectionConfig',
    'M3TMSearchProjection',
    'SearchProjectionFactory',
    'SearchIndexConfig',
    'SearchIndex',
    'SearchIndexFactory',
    'SearchServiceConfig',
    'SearchService',
    'SearchServiceFactory',
    'SearchFilter',
    'Pagination',
    'SearchResult',
    'SearchResults',
    'SearchResultType'
]
