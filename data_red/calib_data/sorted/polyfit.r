library(RcppCNPy)



# Function polynomial
makepoly <- function(form, data, degree = 1) {
  mt <- terms(form, data = data)
  tl <- attr(mt, "term.labels")
  resp <- tl[attr(mt, "response")]
  reformulate(paste0("poly(", tl, ", ", degree, ")"), 
              response = form[[2]])
}


# Function for fitting a bivariate polynomial to a dataset z
polyfit2d <- function(x, y, z, order=4) {
    
    # Define an array G in which to store the polynomial variables (power comb. of x and y)
    Nrrows <- prod(dim(x))
    Nrcols <- (order + 1)^2
    G <- array(0,c(Nrrows,Nrcols))
    
    # Fill G with combinations of powers of x and y
    k <- 0
    for(i in 0:(order+1)){
        for(j in 0:(order+1)){
            G[,k] <- x^i * y^j
            k <- k + 1
        }
        k <- k + 1
    }
    
    # Find a least squares fit of G with respect to the data z
    m <- lsfit(G, z)
    
    return(m)
}



# Load the c- and d-scape numpy files
cscape <- npyLoad("/home/bjung/Documents/Leiden_University/brp/data_red/npsaves/cscape.npy")
dscape <- npyLoad("/home/bjung/Documents/Leiden_University/brp/data_red/npsaves/dscape.npy")

scapex <- seq(ncol(cscape))
scapey <- seq(nrow(cscape))
scapexarcs <- (seq(ncol(cscape)) - median(scapex))*.126
scapeyarcs <- seq(nrow(cscape)) * .126

c_coord <- which(!is.na(cscape), arr.ind=TRUE)
d_coord <- which(!is.na(dscape), arr.ind=TRUE)

c_xarcs <- (c_coord[,2] - median(scapex)) * .126
c_yarcs <- (c_coord[,1]) * .126
d_xarcs <- (d_coord[,2] - median(scapex)) * .126
d_yarcs <- (d_coord[,1]) * .126

cval <- cscape[!is.na(cscape)]
dval <- dscape[!is.na(dscape)]

cdataframe <- data.frame(x=c_xarcs, y=c_yarcs, z=cval)
ddataframe <- data.frame(x=d_xarcs, y=d_yarcs, z=dval)



trsurf <- trmat(polyfit_c, -120, 120, 0, 120, 1000)
eqscplot(trsurf, type = "n")
filled.contour(trsurf, color = terrain.colors,
               plot.title = title(main = "Bivariate third order polynomial fit to c",
                                  xlab = "X [arcsec]", ylab = "Y [arcsec]"),
               plot.axes = { axis(1, seq(-120, 120, by = 20))
                 axis(2, seq(0, 120, by = 20)) },
               key.title = title(main = "c\n[pixels]"),
               key.axes = axis(4, seq(90, 190, by = 10)))  # maybe also asp = 1
points(cdataframe, col = "black")








if(FALSE) {
mpoly_c <- polym(c_xarcs, c_yarcs, degree=3)
polyfit_c <- lm(cval ~ mpoly_c)  
  
persp(scapexarcs, scapeyarcs, trsurf[3], phi = 45, theta = 45,
      xlab = "X Coordinate (feet)", ylab = "Y Coordinate (feet)",
      main = "Surface elevation data")
  
# Create x- and y-grid matrices
scapex <- seq(ncol(cscape))
scapey <- seq(nrow(cscape))
scapexgrid <- matrix(rep(scapex,nrow=length(scapey)))
scapeygrid <- matrix(rep(scapey,ncol=length(scapex)))
# Rescale to arcseconds
scapexarcs <- (scapex - median(scapex))*.126
scapeyarcs <- (scapey - median(scapey))*.126
scapexarcs_grid <- matrix(rep(scapexarcs,nrow=length(scapeyarcs)))
scapeyarcs_grid <- matrix(rep(scapeyarcs,ncol=length(scapexarcs)))

mpoly <- as.function(as.multipol(array(1,c(3,3))))
# Perform least squares bivariate polynomial fit to c- and dscape
polyfit_c <- lm(cscape ~ mpoly)



filled.contour(trsurf, color = terrain.colors,
               plot.title = title(main = "Bivariate third order polynomial fit to c",
                                  xlab = "X [arcsec]", ylab = "Y [arcsec]"),
               xlim=range(-120,120), ylim=range(0,120), 
               key.title = title(main = "c\n[pixels]"),
               key.axes = axis(4, seq(90, 190, by = 10)))  # maybe also asp = 1
points(cdataframe, col = "black", xlim=range(-120,120), ylim=range(0,120))
}










