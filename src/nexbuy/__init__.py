"""
NexBuy recommendation system package
"""

from .models.popular import PopularRecommender
# Other recommenders will be added as they are implemented
# from .models.content_based import ContentBasedRecommender
# from .models.collaborative import CollaborativeRecommender
# from .models.hybrid import HybridRecommender

__all__ = [
    'PopularRecommender',
    # 'ContentBasedRecommender', 
    # 'CollaborativeRecommender',
    # 'HybridRecommender'
]
