# Personalise-an-image-classifier

This project uses VGG16 for transfer learning on CIFAR-10.
Phase 1 trains only the custom classifier head while the base model is frozen.
Phase 2 fine-tunes the top VGG16 layers with a lower learning rate.
The model is evaluated using training and validation accuracy/loss to check performance and overfitting.
