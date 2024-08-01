import numpy as np 
import matplotlib.pyplot as plt
import os,math

def softmax(x):
    """Apply the softmax operation to the input array.

    Args:
        x: The input numpy array, which can be of any shape.

    Returns:
        A numpy array after the softmax operation, with the same shape as the input array. Each element is between 0 and 1, and the sum of all elements in the same row is 1.
    """
    x1 = x - np.max(x, axis = 1, keepdims = True) #减掉最大值防止溢出    
    x1 = np.exp(x1) / np.sum(np.exp(x1), axis = 1, keepdims = True)
    return x1.tolist()

def cosine_similarity(embeddings_1: np.ndarray, embeddings_2: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine similarities between two embedding arrays.

    Args:
        embeddings_1: An array of embeddings of shape (N, D).
        embeddings_2: An array of embeddings of shape (M, D).

    Returns:
        An array of shape (N, M) with the pairwise cosine similarities.
    """

    for embeddings in [embeddings_1, embeddings_2]:
        if len(embeddings.shape) != 2:
            raise ValueError(
                f"Expected 2-D arrays but got shape {embeddings.shape}."
            )

    d1 = embeddings_1.shape[1]
    d2 = embeddings_2.shape[1]
    if d1 != d2:
        raise ValueError(
            "Expected second dimension of embeddings_1 and embeddings_2 to "
            f"match, but got {d1} and {d2} respectively."
        )

    def normalize(embeddings):
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    embeddings_1 = normalize(embeddings_1)
    embeddings_2 = normalize(embeddings_2)
    if d1 == 1024: # CLAP音频embedding计算相似度时需要乘以logits_scale
        logits_scale = 33.3795
        return logits_scale * embeddings_1 @ embeddings_2.T
    else:
        return embeddings_1 @ embeddings_2.T

def get_similarity(embeddings_1: np.ndarray, embeddings_2: np.ndarray,method:str='cosine',use_softmax:bool=True) -> np.ndarray:
    """Compute pairwise similarity scores between two arrays of embeddings.
    Args:
        embeddings_1: An array of embeddings of shape (N, D) or (D,).
        embeddings_2: An array of embeddings of shape (M, D) or (D,).
        method: The method used to compute similarity. Options are 'cosine', 'euclidean', 'manhattan', 'chebyshev', 'pearson'. Default is 'cosine'.
        use_softmax: Whether to apply softmax to the similarity scores. Default is True.

    Returns:
        An array with the pairwise similarity scores. If both inputs are 2-D,
            the output will be of shape (N, M). If one input is 1-D, the output
            will be of shape (N,) or (M,). If both inputs are 1-D, the output
            will be a scalar.
    """
    if embeddings_1.ndim == 1:
        # Convert to 2-D array using x[np.newaxis, :]
        # and remove the extra dimension at the end.
        return get_similarity(
            embeddings_1[np.newaxis, :], embeddings_2
        )[0]

    if embeddings_2.ndim == 1:
        # Convert to 2-D array using x[np.newaxis, :]
        # and remove the extra dimension at the end.
        return get_similarity(
            embeddings_1, embeddings_2[np.newaxis, :]
        )[:, 0]
    if method == 'cosine':
        similarity =  cosine_similarity(embeddings_1, embeddings_2) 
    elif method == 'euclidean':
        distance = np.array([[np.linalg.norm(i - j) for j in embeddings_2] for i in embeddings_1]) 
        sigma = np.mean(distance)  # Or choose sigma in some other way
        similarity = np.exp(-distance ** 2 / (2 * sigma ** 2)) 
    elif method == 'pearson':
        similarity = np.array([[np.corrcoef(i, j)[0,1] for j in embeddings_2] for i in embeddings_1])
    else:
        raise ValueError(
            f"Expected method to be cosine,euclidean and pearson but got {method}."
        )
    if use_softmax:
        return softmax(similarity)
    else:
        return similarity

def visualize_similarity(similarity, x,y,figsize=(10,10)):
    """Visualize the similarity matrix.

    Args:
        similarity: similarity scores matrix. List|ndarray of shape (N, M) or (M, N).
        x: A list of images or texts for each row of the similarity matrix.  List[str]
        y: A list of images or texts for each column of the similarity matrix.  


    Returns:
        A matplotlib figure object.
    """
    # 中文字体，y轴文本/图像
    # plt.rcParams['font.sans-serif']=['times'] #用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

    # 图像尺寸

    plt.figure(figsize=figsize)
    if isinstance(similarity, list):
        similarity = np.array(similarity).T
    else:
        similarity = similarity.T
    if isinstance(x[0], str) and os.path.exists(x[0]):
        x_im = True
        images = [plt.imread(image,0) for image in x]
    else:
        x_im = False
        images = x
    if isinstance(y[0], str) and os.path.exists(y[0]):
        y_im = True
        texts = [plt.imread(image,0) for image in y]
    else:
        y_im = False
        texts = y

    count = len(similarity)
    plt.imshow(similarity, vmin=max(0.0, np.min(similarity)), vmax=np.max(similarity), cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    if x_im and y_im: # x轴和y轴都是图片
        plt.xticks([])
        plt.yticks([])
        for i, image in enumerate(texts):
            plt.imshow(image, extent=( -1.6, -0.6,i + 0.5, i - 0.5,), origin="lower")
        for i, image in enumerate(images):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, count+0.5, count-0.5), origin="lower")
    if y_im and not x_im: # y轴是图片，x轴是文本
        plt.yticks([]) # 去掉y轴刻度
        for i, image in enumerate(texts):
            plt.imshow(image, extent=( -1.6, -0.6,i + 0.5, i - 0.5,), origin="lower")
        plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False,labeltop=True,pad=0)
        plt.xticks(range(len(images)), images,position=(0,1),)#,fontproperties='SimHei')#, fontsize=18)
    if not y_im and x_im: # y轴是文本，x轴是图片
        plt.yticks(range(count), texts)# , fontsize=18)
        plt.xticks([])
        for i, image in enumerate(images):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    if not x_im and not y_im: # x轴和y轴都是文本
        plt.yticks(range(count), texts)# , fontsize=18)
        plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False,labeltop=True,pad=0)
        plt.xticks(range(len(images)), images,position=(0,1),)#,fontproperties='SimHei')#, fontsize=18)

    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.4f}", ha="center", va="center")#, size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)
    if x_im and y_im:
        plt.xlim([-1.6,len(similarity[1]) - 0.5])
        plt.ylim([-0.5, len(similarity) + 0.5])
    elif x_im and not y_im:
        plt.xlim([-0.5, len(similarity[1]) - 0.5])
        plt.ylim([len(similarity)  - 0.5, -1.6])
    elif y_im and not x_im:
        plt.ylim([-0.5, len(similarity) - 0.5])
        plt.xlim([-1.6,len(similarity[1]) - 0.5])
        
    plt.title("Similarity Matrix between Features")
    plt.show()
    return plt

def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        if K < matrix.shape[axis]:
            topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        else:
            topk_index = np.argsort(-matrix, axis=axis)[:, 0:K]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        if K < matrix.shape[axis]:
            topk_index = np.argpartition(-matrix, K, axis=axis)
        else:
            topk_index = np.argsort(-matrix, axis=axis)
        topk_index = topk_index[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort

def find_nearest_square(num):
    next_square = math.ceil(math.sqrt(2*num))
    next_square = next_square + 1 if next_square % 2 != 0 else next_square
    return next_square

def visualize_probability(prob, images, classes, topk=2,figsize=(10,10)):
    """Visualize the probability


    Args:   
        prob: An array of probability scores.
        images: A list of images.
        classes: A list of class names.
        topk: The number of top classes to show. Default is 2.


    Returns:
        A matplotlib figure object.
    """
    plt.figure(figsize=figsize)
    top_probs, top_labels = topk_(np.array(prob), topk)
    if isinstance(images[0], str) and os.path.exists(images[0]):
        images = [plt.imread(image,0) for image in images]

    n = find_nearest_square(len(images))
    for i, image in enumerate(images):
        plt.subplot(n, n, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(n, n, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [classes[index] for index in top_labels[i]])
        # plt.xlabel("probability")
    plt.suptitle(f'Top-{min(topk, len(classes))} probability distribution',y=0.95)
    plt.show()
    return plt



