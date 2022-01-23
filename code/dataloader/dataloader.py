import os
from PIL import Image
from exceptionHandler import raiseException
from torchvision import transforms

ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg", "tif", "tiff"]

class ImageLoader:
	def __init__(self, input_image_path, image_shape):
		self.data_transform = transforms.Compose([
		        transforms.RandomResizedCrop(image_shape),
		        transforms.RandomHorizontalFlip(),
		        transforms.ToTensor(),
		        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		self.input_image_path = input_image_path
		self.image_label = "O"
		self.processed_image = self.processImage()

	def processImage(self):
		image = Image.open(self.input_image_path)
		processed_image = self.data_transform(image).unsqueeze(0)
		return processed_image

def processDocument(document_path):
	processed_image_list = []
	file_extension = document_path.split(".")[-1].lower()
	if(file_extension in ALLOWED_EXTENSIONS):
		processed_image_list.append(document_path)
	return processed_image_list

def processDirectory(input_directory):
	image_file_list = []
	for content in os.listdir(input_directory):
		file_path = os.path.join(input_directory, content)
		if(os.path.isdir(file_path)):
			print(file_path)
			image_file_list += processDirectory(file_path)
		else:
			image_file_list += processDocument(file_path)
	return image_file_list

def processInputImage(input_image_directory, image_shape, logger):
	try:
		image_list = processDirectory(input_image_directory)
		preprocessed_tensor_list = []
		for image_name in image_list:
			image_handler = ImageLoader(image_name, image_shape)
			preprocessed_tensor_list.append(image_handler)
	except:
		logger.exception("Unable to proprocess images")
		raise raiseException("Unable to proprocess images")

	if(len(preprocessed_tensor_list) == 0):
		logger.exception("No Valid Image Found in the Directory " + input_directory)
		raise raiseException("No Valid Image Found in the Directory " + input_directory)
	return preprocessed_tensor_list




