library(rpart)

# it's the most efficient function
# edm3
edm3 <- function(x) {
  x_t_x <- tcrossprod(x)
  d_x_t_x <- diag(x_t_x)
  return(rep(1, length(d_x_t_x)) %*% t(d_x_t_x) - 2 * x_t_x + d_x_t_x %*% t(rep(1, length(d_x_t_x))))
}

# similarity
similarity <- function(x, sigma) {
  return(exp(-edm3(x)/sigma^2))
}

#p_i
p_i <- function(y,
                S,
                H,
                n_u,
                n_l,
                C){
  
  j_nu <- (n_l + 1):(n_l + n_u)
  
  S_i_l <- S[, 1:n_l]
  S_i_u <- S[, j_nu]
  
  p_i_1 <- exp(-2*H) * (S_i_l %*% delta(y[1:n_l], 1))
  p_i_2 <- (C/2) * exp(-H) * (S_i_u %*% exp(H[j_nu]))
  
  return(p_i_1 + p_i_2)
}

# q_i
q_i <- function(y,
                S,
                H,
                n_u,
                n_l,
                C){
  
  j_nu <- (n_l + 1):(n_l + n_u)
  
  S_i_l <- S[, 1:n_l]
  S_i_u <- S[, j_nu]
  
  q_i_1 <- (exp(2*H) * (S_i_l %*% delta(y[1:n_l], -1)))
  q_i_2 <- (C/2) * exp(H) * (S_i_u %*% exp(-H[j_nu]))
  
  return(q_i_1 + q_i_2)
}

# delta
delta <- function(x, y) {
  return(as.integer(x == y))
}


# predict semiboost
predict.semiboost <- function(model, table){
  mod <- model$h
  alpha_mod <- model$alpha
  
  preds_temp  <- lapply(mod, function(x) predict(x, as.data.frame(table), "class"))
  preds <- lapply(preds_temp, function(x) ifelse(as.numeric(x) == 1, 0, 1))
  result <- lapply(preds, function(x) sapply(alpha_mod, function(y) sum(x * y, na.rm=T)))
  
  return(result)
}

## Wheighted Error function 'hi', 'pi' and 'qi'as in formula (12)
w_error <- function(h, p, q) {
  num = 0
  
  num <- sum(if_else(h == -1, num + p, if_else(h == 1, num + q, 0)))
  den <- sum(p + q)
  return(num/den)
}




#' semi-supervised training
#'
#' @description \code{semiboost} fits a semi-supervised training for semiboost model.
#'
#' @param data_train Input \code{matrix} dataset.
#' @param data_test Input \code{matrix} to be predicted.
#' @param target_train Response variable for \code{data_train}. Must be a binary vector and
#' have same length as \code{data_train}.
#' @param target_test Response variable for \code{data_test}. Must be a binary vector and
#' have same length as \code{data_test}.
#' @param sigma A \code{numeric} for the percentile of the distribution of similarities.
#' Default: 4.
#' @param n_iter A \code{numeric} indicating the max number of iterations. Default: 1.
#' @param unlabeled_perc The percentile of unlabeled value samples.


semiboost <- function(data_train, 
                      target_train,
                      
                      data_test, 
                      target_test, 
                      
                      S,
                      
                      n_iter, 
                      unlabeled_perc,
                      
                      print_each = 10,
                      model = "tree"){

	# data_train <- as.matrix(arrange(as_data_frame(data_train), target_train))

	H <- matrix(0, nrow = nrow(data_train), ncol = n_iter + 1)
	n_l <- sum(!is.na(target_train)) # get number of unlabeled
	n_u <- sum(is.na(target_train)) # get number of labeled

	# cat("initializing similarity matrix...", "\n")
	# S <- similarity(data_train, sigma)
	# ca  t("Done!", "\n")

	C <- n_l/n_u
	data_train <- cbind(data_train, labels = 0)
	h <- list()
	alpha_t <- numeric(n_iter + 1)
	error_train <- numeric(n_iter + 1)
	error_test <- numeric(n_iter + 1)
	pred_H_test <- numeric(nrow(data_test))
	unlabeled <- which(!complete.cases(target_train)) # unlabeled observations
	labeled <- which(complete.cases(target_train))

	for (t in 2:(n_iter + 1)){

		 # labeled observations

		p <- p_i(target_train, S, H[ ,t - 1], n_u, n_l, C)
		q <- q_i(target_train, S, H[ ,t - 1], n_u, n_l, C)
		z <- sign(p-q)
		weights <- abs(p-q)[unlabeled]

		P <- weights/ sum(weights)
		sample_unlabeled <- sample(unlabeled, size = unlabeled_perc * length(unlabeled), prob = P)
		data_unlabeled_samples <- data_train[sample_unlabeled,]

		labeled_data <- data_train[complete.cases(target_train),]
		input_data <- rbind(labeled_data, data_unlabeled_samples)

		input_data[ ,"labels"] <- target_train[c(labeled, sample_unlabeled)]
		input_data[is.na(input_data[,"labels"]),  "labels"] <- z[sample_unlabeled]

		input_data[ ,"labels"] <- ifelse(input_data[ ,"labels"] == 1, 1 , 0)

		h[[t]] <-rpart(as.character(labels) ~ .,data = as_data_frame(input_data))#, weights = weights)

		pred_h_train <- as.numeric(as.character(predict(h[[t]], as_data_frame(data_train), type = "class")))
		h_bin_train <- if_else(pred_h_train == 1, 1, -1)
		#vamos a establecer una cota. Los valores mayores de 0.5 seran 1 y los menores 0

		we <- w_error(h_bin_train[unlabeled], p[unlabeled], q[unlabeled])
		alpha_t[t] <- 1/4*log((1-we)/we) ## alfa
		H[ ,t] <- H[ ,t - 1] + (alpha_t[t] * h_bin_train) ## H_t
		error_train[t] <- mean(target_train == sign(H[ ,t]), na.rm = T)

		preds_h_test_temp  <- predict(h[[t]], as_data_frame(data_test), type = "class")
		preds_h_test <- ifelse(as.numeric(as.character(preds_h_test_temp)) == 1, 1, -1)
		pred_H_test <- pred_H_test + (alpha_t[t] * preds_h_test)

		error_test[t] <- mean(target_test == sign(pred_H_test), na.rm = T)

    if ((t - 1) %% print_each == 0)
      cat(paste0("[SemiBoost]", "-iteration", t-1, "  "), " train error =", error_train[t] <- error_train[t],
          paste(" |  test error ="), error_test[t] <- error_test[t],  "\n")
	}

  result <- list(alpha = alpha_t[-1], h = h[-1], error_train = error_train[-1],
					error_test = error_test[-1])
  class(result) <- "semiboost"
  return(result)
}
# 
# model <- semiboost(train_data, test_data, target_train = is_churn, target_test = is_churn_test,
# 					sigma = 1, n_iter = 3, unlabeled_perc = 0.1)
# 
# load("C:/Users/lorena.rodriguez/Documents/SEMIBOOST/semiboost/semiboost")
