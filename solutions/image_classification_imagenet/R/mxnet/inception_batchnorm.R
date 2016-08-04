
require(mxnet)
mx.ctx.internal.default.value = list(device="cpu",device_id=0,device_typeid=1)
class(mx.ctx.internal.default.value) = "MXContext"
require(imager)


#Data gathering
root_path <- c("../../data/Inception/")
model_path <- file.path(root_path,"Inception_BN")
model = mx.model.load(model_path, iteration=39)
mean_path <- file.path(root_path,"mean_224.nd")
mean.img = as.array(mx.nd.load(mean_path)[["mean_img"]])
#Select one image
im <- load.image(system.file("extdata/parrots.png", package="imager"))
#plot(im)

#Image normalization
preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2)
  croped <- crop.borders(im, xx, yy)
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- c(224, 224, 3)
  # substract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}
normed <- preproc.image(im, mean.img)
#plot(normed)

#Predict using the pretrained model
prob <- predict(model, X=normed)
dim(prob)
max.idx <- max.col(t(prob))
max.idx

synsets_path = file.path(root_path,"synset.txt")
synsets <- readLines(synsets_path)
print(paste0("Predicted Top-class: ", synsets[[max.idx]]))


