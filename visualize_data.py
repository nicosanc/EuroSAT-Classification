import matplotlib.pyplot as plt 
from collections import Counter

def visualize_data(dataset):
    num_samples = 5
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    mean= [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    grid = gridspec.GridSpec(num_samples,2)

    for i, idx in enumerate(indices):
      plt.figure(figsize=(10, num_samples*4))
      image, label = dataset[idx]
      original_image = image
      original_image = denormalize_img(original_image, mean, std)
      image = image.permute(1, 2, 0)
      original_image = original_image.permute(1,2,0)

      axis_1 = plt.subplot(grid[i,0])
      axis_1.imshow(original_image)
      axis_1.set_title(label)
      axis_1.axis('off')

      axis_2 = plt.subplot(grid[i,1])
      axis_2.imshow(image)
      axis_2.set_title(label)
      axis_2.axis('off')

      plt.show()

def denormalize_img(image, mean, std):
  image = image.clone()
  for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m).clamp_
  return image

def class_distribution(labels):
    class_count = Counter(labels)
    class_names = list(class_count.keys())
    counts = list(class_count.values())

    plt.pie(counts, labels=class_names, colors=sns.color_palette('pastel'),
            autopct='%.0f%%')

    plt.show()