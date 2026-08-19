library(mgcv)
library(ggsci)

get_partial_sequencing_adjustment <- function()
{
  proportion_of_pcrpositives_sequenced <- read_csv("proportion_of_pcrpositives_sequenced.csv", 
                                                   col_names = FALSE)
  proportion_of_pcrpositives_sequenced$date <- as.Date('2020-12-31') + days(x = seq.int(1,365))
  proportion_of_pcrpositives_sequenced <- proportion_of_pcrpositives_sequenced %>% rename(pct=X1) %>% dplyr::select(date,pct)
  
  # smooth with GAM and/or emwa (try both)
  x        <- matrix(c(1:365,(1:365 %% 7)),ncol=2)
  dat      <- as_tibble(data.frame(y=proportion_of_pcrpositives_sequenced$pct,x1=x[,1],x2=as.factor(x[,2])))
  gam_prop <- gam(y~s(x1,bs='cr',k=50)+s(x2,bs='fs',k=7),data=dat)
  
  gam_pred <- predict(gam_prop,data=dat)
  proportion_of_pcrpositives_sequenced$gam_pred <- gam_pred
  
  lambda <- 0.2
  proportion_of_pcrpositives_sequenced %>% 
    mutate(pct_ewma = accumulate(pct, ~ lambda * .y + (1-lambda)*.x)) %>%
    pivot_longer(-date,values_to = 'pct',names_to = 'smooth') %>%
    ggplot(aes(x=date, y=pct,col=smooth)) + geom_line(aes(linewidth=smooth)) +
    scale_discrete_manual('linewidth',values = c(2,0.5,1))
  
  # read in iar's
  serial.interval <- read_csv('data/serial_interval.csv')
  si_short        <- c(serial.interval$fit[1:29],serial.interval$fit[30:100] |> sum())
  rt_iar_file     <- read_csv('data/denmark_rt_iar.csv') %>% 
    mutate(Rc    = rollapply(Rt_adj,width=30,w=si_short,FUN=weighted.mean,fill=NA,align = 'left'),
           Rc_lb = rollapply(Rt_adj_0.025,width=30,w=si_short,FUN=weighted.mean,fill=NA,align = 'left'),
           Rc_ub = rollapply(Rt_adj_0.975,width=30,w=si_short,FUN=weighted.mean,fill=NA,align = 'left'),
           iar_shifted = rollapply(iar_0.5,width=30,w=si_short,FUN=weighted.mean,fill=NA,align = 'left'),
           iar_lb_shifted = rollapply(iar_0.025,width=30,w=si_short,FUN=weighted.mean,fill=NA,align = 'left'),
           iar_ub_shifted = rollapply(iar_0.975,width=30,w=si_short,FUN=weighted.mean,fill=NA,align = 'left'))
  
  rt_iar_file <- rt_iar_file %>% left_join(proportion_of_pcrpositives_sequenced,by=c('date')) %>%
    mutate(gam_pred_adj = replace_na(gam_pred,1),
           partial_sequencing = gam_pred_adj*iar_0.5,
           partial_sequencing_shifted = gam_pred_adj*iar_shifted)
  
  return(rt_iar_file)
}



