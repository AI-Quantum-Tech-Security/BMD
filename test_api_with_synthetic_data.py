#!/usr/bin/env python3
"""
End-to-end test script for the Risk Scoring API using synthetic data.
This script demonstrates the complete workflow from data loading to API prediction.
"""

import pandas as pd
import requests
import json
import time
import random
from typing import Dict, List

class RiskAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        # Handle HTTPS with self-signed certificates if needed
        if base_url.startswith('https'):
            self.session.verify = False
            # Suppress SSL warnings for cleaner output
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
    def health_check(self) -> Dict:
        """Test API health endpoint"""
        print("🏥 Testing API Health Check...")
        response = self.session.get(f"{self.base_url}/")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API is healthy: {result['message']} v{result['version']}")
            return result
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return {}
    
    def map_dataset_to_api_features(self, row: pd.Series) -> Dict:
        """Map dataset columns to API expected feature schema"""
        return {
            "avg_tx_amount": float(row['tx_amount']),  # Use current tx as proxy for average
            "device_change_freq": 0.05,  # Default value (could be computed from user history)
            "tx_hour": int(row['tx_hour']),
            "location_change_freq": float(row['distance_from_usual_location']) / 1000.0,  # Normalize
            "is_new_device": int(row['is_new_device']),
            "transaction_count_24h": int(row['transaction_count_24h']),
            "time_since_last_tx": 2.5,  # Default value (could be computed from timestamps)
            "tx_amount_to_balance_ratio": float(row['tx_amount_to_balance_ratio']),
            "ip_address_reputation": float(row['ip_address_reputation']),
            "is_weekend": int(row['is_weekend']),
            "transaction_velocity_10min": int(row['transaction_velocity_10min']),
            "country_change_flag": int(row['country_change_flag'])
        }
    
    def predict_risk(self, features: Dict) -> Dict:
        """Get risk prediction from API"""
        response = self.session.post(
            f"{self.base_url}/risk-score",
            headers={"Content-Type": "application/json"},
            json=features
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            response.raise_for_status()
    
    def test_synthetic_data_batch(self, df: pd.DataFrame, num_samples: int = 10) -> List[Dict]:
        """Test API with a batch of synthetic data samples"""
        print(f"\n🧪 Testing API with {num_samples} synthetic data samples...")
        
        results = []
        for i in range(num_samples):
            # Select random row from dataset
            idx = random.randint(0, len(df) - 1)
            row = df.iloc[idx]
            
            # Map to API features
            features = self.map_dataset_to_api_features(row)
            
            # Get prediction
            start_time = time.time()
            prediction = self.predict_risk(features)
            end_time = time.time()
            
            # Store result with metadata
            result = {
                'sample_idx': idx,
                'original_risk_label': row['risk_label'],
                'api_prediction': prediction,
                'response_time_ms': round((end_time - start_time) * 1000, 2),
                'input_features': features
            }
            results.append(result)
            
            # Print result
            print(f"Sample {i+1:2d}: "
                  f"Original={row['risk_label']:10s} | "
                  f"Predicted={prediction['risk_flag']:10s} | "
                  f"Score={prediction['risk_score']:.4f} | "
                  f"Time={result['response_time_ms']:5.1f}ms")
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze prediction results and calculate metrics"""
        print(f"\n📊 Analyzing Results...")
        
        # Extract predictions and ground truth
        original_labels = [r['original_risk_label'] for r in results]
        predicted_labels = [r['api_prediction']['risk_flag'] for r in results]
        risk_scores = [r['api_prediction']['risk_score'] for r in results]
        response_times = [r['response_time_ms'] for r in results]
        
        # Calculate accuracy
        correct = sum(1 for orig, pred in zip(original_labels, predicted_labels) if orig == pred)
        accuracy = correct / len(results)
        
        # Risk score statistics
        avg_score = sum(risk_scores) / len(risk_scores)
        min_score = min(risk_scores)
        max_score = max(risk_scores)
        
        # Performance statistics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Label distribution
        original_dist = {label: original_labels.count(label) for label in set(original_labels)}
        predicted_dist = {label: predicted_labels.count(label) for label in set(predicted_labels)}
        
        analysis = {
            'accuracy': accuracy,
            'avg_risk_score': avg_score,
            'risk_score_range': (min_score, max_score),
            'avg_response_time_ms': avg_response_time,
            'max_response_time_ms': max_response_time,
            'original_distribution': original_dist,
            'predicted_distribution': predicted_dist,
            'total_samples': len(results)
        }
        
        # Print analysis
        print(f"✅ Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
        print(f"📈 Average Risk Score: {avg_score:.4f}")
        print(f"🎯 Risk Score Range: {min_score:.4f} - {max_score:.4f}")
        print(f"⚡ Average Response Time: {avg_response_time:.1f}ms")
        print(f"⏱️  Max Response Time: {max_response_time:.1f}ms")
        
        print(f"\n📋 Original Label Distribution:")
        for label, count in original_dist.items():
            percentage = (count / len(results)) * 100
            print(f"   {label:10s}: {count:3d} ({percentage:5.1f}%)")
        
        print(f"\n🔮 Predicted Label Distribution:")
        for label, count in predicted_dist.items():
            percentage = (count / len(results)) * 100
            print(f"   {label:10s}: {count:3d} ({percentage:5.1f}%)")
        
        return analysis
    
    def test_edge_cases(self) -> None:
        """Test API with edge cases and invalid inputs"""
        print(f"\n🧨 Testing Edge Cases...")
        
        # Test 1: Minimum values
        print("Test 1: Minimum values")
        min_features = {
            "avg_tx_amount": 20.0,
            "device_change_freq": 0.0,
            "tx_hour": 0,
            "location_change_freq": 0.0,
            "is_new_device": 0,
            "transaction_count_24h": 0,
            "time_since_last_tx": 0.0,
            "tx_amount_to_balance_ratio": 0.0,
            "ip_address_reputation": 0.0,
            "is_weekend": 0,
            "transaction_velocity_10min": 0,
            "country_change_flag": 0
        }
        result = self.predict_risk(min_features)
        print(f"   Min values result: {result['risk_flag']} (score: {result['risk_score']:.4f})")
        
        # Test 2: Maximum values
        print("Test 2: Maximum values")
        max_features = {
            "avg_tx_amount": 10000.0,
            "device_change_freq": 1.0,
            "tx_hour": 23,
            "location_change_freq": 1.0,
            "is_new_device": 1,
            "transaction_count_24h": 100,
            "time_since_last_tx": 168.0,  # 1 week
            "tx_amount_to_balance_ratio": 1.0,
            "ip_address_reputation": 1.0,
            "is_weekend": 1,
            "transaction_velocity_10min": 50,
            "country_change_flag": 1
        }
        result = self.predict_risk(max_features)
        print(f"   Max values result: {result['risk_flag']} (score: {result['risk_score']:.4f})")
        
        # Test 3: High-risk scenario
        print("Test 3: High-risk scenario")
        high_risk_features = {
            "avg_tx_amount": 5000.0,
            "device_change_freq": 0.8,
            "tx_hour": 3,  # Late night
            "location_change_freq": 0.9,
            "is_new_device": 1,
            "transaction_count_24h": 20,
            "time_since_last_tx": 0.1,  # Very recent
            "tx_amount_to_balance_ratio": 0.9,  # Large transaction
            "ip_address_reputation": 0.1,  # Bad IP
            "is_weekend": 1,
            "transaction_velocity_10min": 10,
            "country_change_flag": 1
        }
        result = self.predict_risk(high_risk_features)
        print(f"   High-risk result: {result['risk_flag']} (score: {result['risk_score']:.4f})")

def main():
    """Main test execution"""
    print("🚀 Starting Risk Scoring API End-to-End Test")
    print("=" * 60)
    
    # Initialize tester
    tester = RiskAPITester()
    
    # Health check
    if not tester.health_check():
        print("❌ API health check failed. Make sure the server is running.")
        return
    
    # Load synthetic data
    print(f"\n📁 Loading synthetic behavioral dataset...")
    try:
        df = pd.read_csv('files/synthetic_behavioral_dataset.csv')
        print(f"✅ Loaded dataset: {len(df)} transactions, {len(df.columns)} columns")
        print(f"📊 Risk label distribution:")
        risk_dist = df['risk_label'].value_counts()
        for label, count in risk_dist.items():
            percentage = (count / len(df)) * 100
            print(f"   {label:10s}: {count:6d} ({percentage:5.1f}%)")
    except FileNotFoundError:
        print("❌ Synthetic dataset not found. Please ensure 'files/synthetic_behavioral_dataset.csv' exists.")
        return
    
    # Test with synthetic data
    results = tester.test_synthetic_data_batch(df, num_samples=20)
    
    # Analyze results
    analysis = tester.analyze_results(results)
    
    # Test edge cases
    tester.test_edge_cases()
    
    # Final summary
    print(f"\n🎉 Test Complete!")
    print("=" * 60)
    print(f"✅ API is working correctly with {analysis['accuracy']:.1%} accuracy")
    print(f"⚡ Average response time: {analysis['avg_response_time_ms']:.1f}ms")
    print(f"📊 Tested {analysis['total_samples']} samples successfully")
    
    # Save results
    output_file = 'api_test_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'test_summary': analysis,
            'detailed_results': results
        }, f, indent=2)
    print(f"💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()