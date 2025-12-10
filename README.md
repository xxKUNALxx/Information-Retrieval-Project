
PROJECT DEMO-- https://drive.google.com/file/d/1toZlhf_f-v-Wzf9VqjnryBlCwxhPJgOF/view?usp=sharing

# Fashion Recommendation System 👗🛍️

An advanced machine learning-powered fashion recommendation system that predicts customer purchases for H&M using hybrid collaborative filtering and customer segmentation techniques.

## 🌟 Project Overview

This project addresses the challenge of choice overload in fashion e-commerce by delivering highly personalized product recommendations. Unlike traditional recommendation systems that rely solely on purchase history, our solution combines multiple data-driven techniques to enhance accuracy and user experience.

## ✨ Key Features

- **Hybrid Collaborative Filtering**: Combines Singular Value Decomposition (SVD) and Alternating Least Squares (ALS) algorithms
- **Customer Segmentation**: K-Means clustering for targeted recommendations based on purchasing behavior
- **RFM Analysis**: Recency, Frequency, and Monetary value analysis for customer categorization
- **Comprehensive EDA**: Deep insights into customer behavior, seasonal trends, and product popularity
- **Cold Start Problem Handling**: Effective recommendations for new users and products
- **Scalable Architecture**: Designed to handle large datasets efficiently

## 🔧 Technical Stack

- **Programming Language**: Python
- **Machine Learning**: scikit-learn, PyTorch
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Clustering**: K-Means, PCA for dimensionality reduction
- **Matrix Factorization**: SVD, ALS with regularization
- **GPU Acceleration**: CUDA-enabled training

## 📊 Dataset

The project utilizes the **H&M Personalized Fashion Recommendations** dataset from Kaggle, containing:
- Customer transaction history
- Product metadata (categories, prices, descriptions)
- Customer demographics
- Article images and attributes

## 🚀 Key Innovations

1. **Multi-faceted Approach**: Integrates collaborative filtering with customer segmentation for enhanced personalization
2. **Feature Engineering**: Time-based features, price sensitivity analysis, and repeat purchase behavior
3. **Advanced Clustering**: Customer segmentation using RFM analysis and K-Means clustering
4. **Performance Optimization**: GPU-accelerated training with regularization techniques

## 📈 Model Performance

| Model | Precision@10 | Recall@10 | MSE | Speed |
|-------|-------------|-----------|-----|-------|
| SVD   | 0.72        | 0.68      | 0.015 | Fast |
| ALS   | 0.78        | 0.74      | 0.012 | Moderate |

**Best Performing Model**: ALS achieved superior accuracy with better handling of sparse data and higher personalization capabilities.

## 🔍 Key Insights from EDA

- **Top Product Categories**: Trousers, T-shirts, and socks dominate purchases
- **Customer Segments**: Identified distinct groups including high-value shoppers, occasional buyers, and price-sensitive customers  
- **Seasonal Trends**: Transaction patterns show periodic spikes during promotional periods
- **Channel Distribution**: Analysis of online vs. in-store purchasing behavior

## 🎯 Business Impact

- **Enhanced User Experience**: Personalized recommendations reduce choice overload
- **Increased Conversion Rates**: Targeted suggestions improve sales performance
- **Inventory Optimization**: Insights into product popularity aid inventory management
- **Customer Retention**: Segmentation enables targeted marketing strategies

## 🔮 Future Enhancements

- [ ] **Deep Learning Integration**: Implement Transformer-based models (BERT4Rec, SASRec)
- [ ] **Real-time Recommendations**: Session-based dynamic adaptation
- [ ] **Hybrid Approach**: Combine content-based and collaborative filtering
- [ ] **Image Recognition**: Visual similarity recommendations using CNNs
- [ ] **A/B Testing Framework**: Continuous model improvement and validation

## 📊 Evaluation Metrics

- **Precision@K**: Relevance of top-K recommendations
- **Recall@K**: Coverage of relevant items
- **Mean Squared Error (MSE)**: Prediction accuracy
- **Silhouette Score**: Clustering quality assessment



## 🏫 Institution

**SRM University-AP**  
Department of Computer Science & Engineering  
Mangalagiri, Guntur, Andhra Pradesh - 522502

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- H&M for providing the comprehensive dataset
- Kaggle community for dataset hosting and resources
- SRM University-AP for academic support and guidance
- PyTorch and scikit-learn communities for excellent documentation

## 📚 References

- [H&M Kaggle Competition](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/data)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

⭐ **Star this repository if you found it helpful!** ⭐
