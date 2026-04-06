
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_config = {'lr': lr, 'filters': f}
            print(f"⭐ New Fittest Individual Found! Accuracy: {acc:.4f}")

print("\n--- OPTIMIZATION COMPLETE ---")
print(f"Best Fitness Score: {best_accuracy:.4f}")
print(f"Optimal Genotype (Config): {best_config}")
