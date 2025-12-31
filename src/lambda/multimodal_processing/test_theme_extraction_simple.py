#!/usr/bin/env python3
"""
Simple test script for enhanced theme extraction implementation.
"""

import sys
import os

# Test TF-IDF clustering directly
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn is available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ scikit-learn not available, using fallback")

def preprocess_text(text):
    """Preprocess text for TF-IDF vectorization."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def test_tfidf_clustering():
    """Test TF-IDF clustering with sample comments."""
    
    # Sample survey comments for testing
    test_comments = [
        "The product quality is excellent and design is very intuitive",
        "Customer service was very helpful and responsive to my issues",
        "The price is reasonable for the value provided",
        "Delivery was fast and the package arrived in good condition",
        "The user interface is easy to navigate but some features are confusing",
        "Had some technical issues with the latest update",
        "Billing was straightforward but the payment process was slow",
        "Overall satisfaction is high with the product features"
    ]
    
    print("Testing TF-IDF clustering...")
    print(f"Input comments: {len(test_comments)}")
    
    if not SKLEARN_AVAILABLE:
        print("❌ Cannot test TF-IDF clustering without scikit-learn")
        return False
    
    # Preprocess comments
    processed_comments = [preprocess_text(comment) for comment in test_comments]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.8  # Ignore terms that appear in more than 80% of documents
    )
    
    # Fit and transform comments
    tfidf_matrix = vectorizer.fit_transform(processed_comments)
    
    # Determine optimal number of clusters using silhouette score
    max_clusters = min(8, len(test_comments) // 2)  # Cap at 8 clusters
    if max_clusters < 2:
        max_clusters = 2
        
    best_score = -1
    best_k = 2
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, cluster_labels)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Optimal number of clusters: {best_k} (silhouette score: {best_score:.3f})")
    
    # Apply K-means with optimal number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    # Extract themes from clusters
    feature_names = vectorizer.get_feature_names_out()
    themes = []
    
    for cluster_id in range(best_k):
        # Get comments in this cluster
        cluster_comments = [test_comments[i] for i in range(len(test_comments)) if cluster_labels[i] == cluster_id]
        
        # Get top terms for this cluster
        center = kmeans.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-5:][::-1]  # Get top 5 terms
        top_terms = [feature_names[i] for i in top_indices]
        
        themes.append({
            'cluster_id': cluster_id,
            'comment_count': len(cluster_comments),
            'top_terms': top_terms,
            'sample_comments': cluster_comments[:2]  # Show first 2 comments as examples
        })
    
    # Sort themes by comment count (most common first)
    themes.sort(key=lambda x: x['comment_count'], reverse=True)
    
    print(f"Extracted {len(themes)} themes:")
    for i, theme in enumerate(themes[:5]):
        print(f"  {i+1}. {theme['top_terms'][:3]} ({theme['comment_count']} comments)")
        print(f"     Sample: {theme['sample_comments'][0][:50]}...")
    
    return True

if __name__ == "__main__":
    success = test_tfidf_clustering()
    if success:
        print("✅ TF-IDF clustering test successful")
    else:
        print("❌ TF-IDF clustering test failed")