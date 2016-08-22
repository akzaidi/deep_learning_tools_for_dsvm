# function to convert vector to matrix
vec2mat <- function(x, nrow=vocab_size, ncol=feature_len){
  m <- matrix(0L, nrow, ncol)
  m[cbind(x, 1:ncol)] <- 1L
  m
}

# Custom CSVIter based on mx.io.CSVIter 
CustomCSVIter <- function (data.csv, data.shape, batch.size, shuffle=FALSE) 
{
  vocab_size <- data.shape[1]
  feature_len <- data.shape[2]
  csv_iter <- mx.io.CSVIter(data.csv=data.csv, data.shape=c(1,feature_len), 
                            batch.size=batch.size)
  csv_iter$iter.next()
  vect <- as.array(csv_iter$value()$data)
  vect <- trunc(vect)
  # array dimension: it's width, height, channels, samples
  dat <- array(NA, dim=c(vocab_size, feature_len, 1, batch.size))
  for(b in 1:batch.size){
    dat[, ,1 , b] <- vec2mat(vect[, , b],vocab_size, feature_len)
  }
  dim(dat) <- c(vocab_size, feature_len, 1, batch.size)
  label <- array(0, c(1,batch.size))#temp
  custom_iter <- mx.io.arrayiter(data=dat, label=label, batch.size=batch.size, 
                                 shuffle=shuffle) 
}

