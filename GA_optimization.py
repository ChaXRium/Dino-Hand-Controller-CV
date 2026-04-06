for lr in learning_rates:
    for f in filter_sizes:
        print(f"Testing Individual -> LR: {lr}, Filters: {f}...")
        model = create_individual(lr, f)
        
        # Fitness Test (2 epochs is enough to see potential)
        history = model.fit(train_ds, epochs=2, verbose=0) 
        acc = max(history.history['accuracy'])
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_config = {'lr': lr, 'filters': f}
            print(f"⭐ New Fittest Individual Found! Accuracy: {acc:.4f}")

print("\n--- OPTIMIZATION COMPLETE ---")
print(f"Best Fitness Score: {best_accuracy:.4f}")
print(f"Optimal Genotype (Config): {best_config}")
