1. Environment Setup
Students are strongly advised to use Python 3.7 for this assginment. Using other versions may lead to environment and testing issues.

If you encounter minor environment problems (e.g., missing packages), these can typically be resolved by installing the required dependencies via Conda, for example: conda install <package_name>.

2. RNN_Captioning.ipynb
```py
    # Sample a minibatch and show the images and captions.
    # If you get an error, the URL just no longer exists, so don't worry! # You can re-sample as many times as you want.
    batch_size = 3


    captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size) 
    for i, (caption, url) in enumerate(zip(captions, urls)):
        plt.imshow(image_from_url(url)) 
        plt.axis('off')
        caption_str = decode_captions(caption, data['idx_to_word'])
        plt.title(caption_str) 
        plt.show()
```
Some url already does not exist. You can run it several time until success.

3. collect_submission.ipynb
You need to run all the notebooks before running collect_submission.ipynb to ensure that all necessary files are generated for submission. If you encounter issues in this process, try using the following command:

%cd /content/drive/My\ Drive/$FOLDERNAME
!sudo apt-get update
!sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain- generic pandoc
!pip install PyPDF2
!bash collectSubmission.sh