library(tidyverse)
library(zoo)

source('partial_sequencing_adjustment.R')

which_min <- function(x, pos) {
  which(x==sort(unique(x))[pos])
}

getVariant_fromString <- function(x)
{
  y   <- str_split(x,':')
  out <- rep('',length(y))
  
  for(i in 1:length(y))
  {
    for(el in y[[i]])
      if(str_detect(el,'variant'))
        out[i] <- str_replace(el,'variant','')
  }
  
  return(out)
}

getNPI_fromString <- function(x)
{
  y   <- str_split(x,':')
  out <- rep('',length(y))
  
  for(i in 1:length(y))
  {
    for(el in y[[i]])
      if(str_detect(el,'_combined_numeric')|str_detect(el,'as.factor[(]'))
        out[i] <- str_replace(str_replace(str_replace(el,'_combined_numeric',''),'as.factor[(]',''),'[)]','_')
  }
   
  return(out)  
}

getNPI_name <- function(x,NPI)
{
  y   <- str_detect(x,'_')
  out <- rep(NPI,length(y))
  
  for(i in 1:length(y))
  {
    if(y[i]) out[i] <- paste0(NPI,'_',str_split(x[i],'_')[[1]][2])
  }
  
  return(out)
}

get_onward_work_infection_fromString <- function(x)
{
  y   <- str_split(x,':')
  out <- rep(FALSE,length(y))
  
  for(i in 1:length(y))
  {
    for(el in y[[i]])
      if(str_detect(el,'onward_work_infection'))
        out[i] <- as.logical(str_replace(el,'onward_work_infection',''))
  }
  
  return(out)  
}

get_Infector_share_workplace_fromString <- function(x)
{
  y   <- str_split(x,':')
  out <- rep(FALSE,length(y))
  
  for(i in 1:length(y))
  {
    for(el in y[[i]])
      if(str_detect(el,'Infector_share_workplace'))
        out[i] <- as.logical(str_replace(el,'Infector_share_workplace',''))
  }
  
  return(out)  
}

get_onward_school_infection_fromString <- function(x)
{
  y   <- str_split(x,':')
  out <- rep(FALSE,length(y))
  
  for(i in 1:length(y))
  {
    for(el in y[[i]])
      if(str_detect(el,'onward_school_infection'))
        out[i] <- as.logical(str_replace(el,'onward_school_infection',''))
  }
  
  return(out)  
}

get_Infector_share_school_fromString <- function(x)
{
  y   <- str_split(x,':')
  out <- rep(FALSE,length(y))
  
  for(i in 1:length(y))
  {
    for(el in y[[i]])
      if(str_detect(el,'Infector_share_school'))
        out[i] <- as.logical(str_replace(el,'Infector_share_school',''))
  }
  
  return(out)  
}


get_combined_covid_data_Denmark <- function(
    use_saved_data   = FALSE,
    file_name        = ''
)
{
  if(use_saved_data & file.exists(file_name)) return( readRDS(file_name) )
  
  # OxCGRT
  covid_oxcgrt          <- read_csv('https://github.com/OxCGRT/covid-policy-tracker-legacy/blob/main/legacy_data_202207/OxCGRT_latest_combined.csv?raw=true') %>%
    mutate(Date=as.Date(as.character(Date),"%Y%m%d")) %>% filter(CountryName=="Denmark")
  covid_vaccination     <- read_csv('https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv?raw=true') %>%
    filter(location=='Denmark')
  
  # Data on Covid variants
  covariant_org_data    <- readRDS(paste0("NPI_behaviour_data/covariant_org_data.rds")) %>% filter(iso3c=='DNK')
  cod_colnames          <- colnames(covariant_org_data)
  cod_colnames[cod_colnames!="date"&cod_colnames!="iso3c"] <- paste0("SARS_CoV_2_Variant_",cod_colnames[cod_colnames!="date"&cod_colnames!="iso3c"])
  colnames(covariant_org_data) <- cod_colnames
  
  dominant_variant <- covariant_org_data %>%
    rowwise() %>% dplyr::select(-c(date,iso3c)) %>%
    mutate(dominant_variant = names(cur_data())[which.max(c_across(everything()))]) %>%
    pull("dominant_variant")
  
  covariant_org_data$SARS_CoV_2_Dominant_Variant <- dominant_variant
  
  # Construct new variable
  covariant_org_data$WT_variant      <- 1    # this is always 1
  covariant_org_data$Alpha_variant   <- 0    # initialize the others with zeros
  covariant_org_data$Delta_variant   <- 0
  covariant_org_data$Omicron_variant <- 0
  
  covariant_org_data[covariant_org_data$SARS_CoV_2_Dominant_Variant!="SARS_CoV_2_Variant_Other",]$Alpha_variant <- 1   # if not WT set Alpha to 1
  covariant_org_data[covariant_org_data$SARS_CoV_2_Dominant_Variant!="SARS_CoV_2_Variant_Other"&covariant_org_data$SARS_CoV_2_Dominant_Variant!="SARS_CoV_2_Variant_Alpha",]$Delta_variant <- 1
  covariant_org_data[covariant_org_data$SARS_CoV_2_Dominant_Variant!="SARS_CoV_2_Variant_Other"&covariant_org_data$SARS_CoV_2_Dominant_Variant!="SARS_CoV_2_Variant_Alpha"&covariant_org_data$SARS_CoV_2_Dominant_Variant!="SARS_CoV_2_Variant_Delta",]$Omicron_variant <- 1
  
  # Google Mobility data
  mobility_cnt_level <- readRDS(paste0("NPI_behaviour_data/google_mobility_cnt.rds")) %>% filter(iso3c=='DNK')
  
  # Mortality stats (...)
  Economist_exDeath <- read_csv("https://github.com/TheEconomist/covid-19-the-economist-global-excess-deaths-model/blob/main/output-data/export_country_per_100k.csv?raw=true") %>% filter(iso3c=='DNK')
  
  # Combine all the data sets
  covid_data <- covid_oxcgrt %>%
    left_join(covid_vaccination,by=c("CountryCode"="iso_code","Date"="date")) %>%
    left_join(covariant_org_data,by=c('CountryCode'='iso3c','Date'='date')) %>%
    left_join(mobility_cnt_level,by=c("CountryCode"="iso3c","Date"="date")) %>%
    left_join(Economist_exDeath,by=c("CountryCode"="iso3c","Date"="date"))
  
  write_rds(covid_data,file = file_name)
  
  return(covid_data)
}

#' Title YouGov Imperial Covid behavioural tracker study.
#'
#' @return    data frame with behavioural survey data for denmark
get_DNK_data <- function()
{
  covid_data  <- get_combined_covid_data_Denmark( use_saved_data = TRUE,
                                                  file_name      = 'data/NPI_behaviour_data/oxcgrt_npi_data.rds')
  
  
  yougov_dnk     <- readRDS('data/NPI_behaviour_data/yougov_dnk.rds')
  yougov_dnk_agg <- yougov_dnk %>% group_by(qweek) %>%
    summarise(n                  = n(),
              date               = max(date),    # take the last date of the qweek
              avg_age            = mean(age),
              avg_hsize          = mean(i1_health),
              wgt_hsize          = weighted.mean(i1_health,weight,na.rm=TRUE),
              avg_contact        = mean(i2_health),
              wgt_contact        = weighted.mean(i2_health,weight,na.rm=TRUE),
              avg_lefthouse      = mean(i7a_health),
              wgt_lefthouse      = weighted.mean(i7a_health,weight,na.rm=TRUE),
              avg_wouldisolate   = mean(i9_health),
              wgt_wouldisolate   = weighted.mean(i9_health,weight,na.rm=TRUE),
              avg_easetoisolate  = mean(i10_health),
              wgt_easetoisolate  = weighted.mean(i10_health,weight,na.rm=TRUE),
              avg_adviseisolate  = mean(i11_health),
              wgt_adviseisolate  = weighted.mean(i11_health,weight,na.rm=TRUE),
              avg_facemask       = mean(i12_health_1),
              wgt_facemask       = weighted.mean(i12_health_1,weight,na.rm=TRUE),
              avg_handwashing    = mean(i12_health_2),
              wgt_handwashing    = weighted.mean(i12_health_2,weight,na.rm=TRUE),
              avg_avoidout       = mean(i12_health_6),
              wgt_avoidout       = weighted.mean(i12_health_6,weight,na.rm=TRUE),
              avg_avoidhospital  = mean(i12_health_7),
              wgt_avoidhospital  = weighted.mean(i12_health_7,weight,na.rm=TRUE),
              avg_avoidtransport = mean(i12_health_8),
              wgt_avoidtransport = weighted.mean(i12_health_8,weight,na.rm=TRUE),
              avg_avoidoutsidew  = mean(i12_health_9),
              wgt_avoidoutsidew  = weighted.mean(i12_health_9,weight,na.rm=TRUE),
              avg_houseguests    = mean(i12_health_11),
              wgt_houseguests    = weighted.mean(i12_health_11,weight,na.rm=TRUE),
              avg_smallgather    = mean(i12_health_12),
              wgt_smallgather    = weighted.mean(i12_health_12,weight,na.rm=TRUE),
              avg_mediumgather   = mean(i12_health_13),
              wgt_mediumgather   = weighted.mean(i12_health_13,weight,na.rm=TRUE),
              avg_largegather    = mean(i12_health_14),
              wgt_largegather    = weighted.mean(i12_health_14,weight,na.rm=TRUE),
              avg_avoidcrowds    = mean(i12_health_15),
              wgt_avoidcrowds    = weighted.mean(i12_health_15,weight,na.rm=TRUE),
              avg_avoidshops     = mean(i12_health_16),
              wgt_avoidshops     = weighted.mean(i12_health_16,weight,na.rm=TRUE),
              avg_govhandling    = mean(WCRex1),
              wgt_govhandling    = weighted.mean(WCRex1,weight,na.rm=TRUE),
              #avg_confidenceHS   = mean(WCRex2),
              #wgt_confidenceHS   = weighted.mean(WCRex2,weight),
              avg_govtrust       = mean(ox5_2),
              wgt_govtrust       = weighted.mean(ox5_2,weight,na.rm=TRUE),
              avg_lifesatisfact  = mean(cantril_ladder),
              wgt_lifesatisfact  = weighted.mean(cantril_ladder,weight,na.rm=TRUE),
              avg_longcovid      = mean(long_covid),
              wgt_longcovid      = weighted.mean(long_covid,weight,na.rm=TRUE)) %>% ungroup()
  
  joined_data <- covid_data %>% left_join(yougov_dnk_agg,by=c('Date'='date'))
  
  joined_data <- joined_data %>% arrange(Date) %>%
    fill(`transit stations`) %>%
    mutate(transit_week_mean = rollapply(`transit stations`,width=7,mean,align="right",fill=NA),
           ConfirmedCases = replace_na(ConfirmedCases,0),
           ConfirmedDeaths = replace_na(ConfirmedDeaths,0),
           Cases  = rollapply((ConfirmedCases - dplyr::lag(ConfirmedCases,1))/1000,width=7,mean,align="right",fill=NA),
           Deaths = rollapply((ConfirmedDeaths - dplyr::lag(ConfirmedDeaths,1))/10,width=7,mean,align="right",fill=NA),
           H6     = H6_combined_numeric/4*5)
  
  joined_data$tStringency <- NA
  first_date_Stringency   <- joined_data %>% filter(StringencyIndex>10) %>% summarize(First_StringencyIndex = min(Date)) %>%
    ungroup() %>% summarise(First_StringencyIndex=mean(First_StringencyIndex))# use Stringency Index 10 to remove early noise
  
  #joined_data <- left_join(joined_data,first_date_Stringency)
  joined_data$First_StringencyIndex <- first_date_Stringency$First_StringencyIndex #first_date_Stringency$First_StringencyIndex
  joined_data <- joined_data %>% filter(Date < as.Date('2021-12-01')) %>%
    mutate(tStringency   = replace(as.numeric(pmax((Date - First_StringencyIndex)/30,0)),as.numeric(pmax((Date - First_StringencyIndex)/30,0))==0,NA),
           tStringency   = tStringency - mean(na.omit(tStringency)),
           tStringencySq = tStringency^2)
  
  joined_data <- joined_data %>% mutate(C1=5*C1_combined_numeric/3,
                                        C2=5*C2_combined_numeric/3,
                                        C3=5*C3_combined_numeric/2,
                                        C4=5*C4_combined_numeric/4,
                                        C5=5*C5_combined_numeric/2,
                                        C6=5*C6_combined_numeric/3,
                                        C7=5*C7_combined_numeric/2,
                                        C8=5*C8_combined_numeric/4,
                                        H6=5*H6_combined_numeric/4,
                                        H8=5*H6_combined_numeric/3,
                                        avoid_gatherings = (wgt_houseguests + wgt_smallgather + wgt_mediumgather + wgt_largegather + wgt_avoidshops)/5,
                                        stay_at_home     = (wgt_avoidout + wgt_avoidcrowds)/2,
                                        #vacc_per_person  = replace_na(Admin_Per_100K,0)/100000
  )
  
  return(joined_data)
}


if(FALSE)
{
  joined_data <- get_DNK_data()
  
  colors <- c("Mask Survey" = "blue", "Mask Mandate" = "red", "Transit [inverse/4]" = "darkgreen", "R_t"="pink",
              "Cases (thousand)" = "black", "Deaths (ten)" = "brown")
  
  # stay at home, avoid gatherings
  
  mask_plot <- joined_data %>% filter(Date>as.Date('2020-10-01')&Date<as.Date('2021-12-31')) %>%
    ggplot(aes(x=Date)) + geom_point(aes(y=wgt_facemask,color="Mask Survey")) +
    geom_step(aes(y=H6,color="Mask Mandate")) +
    geom_line(aes(y=-transit_week_mean/25,color="Transit [inverse/4]")) +
    geom_line(aes(y=Cases,color="Cases (thousand)")) +
    geom_line(aes(y=Deaths,color="Deaths (ten)")) +
    ggtitle("Facial covering") + labs(x = "",y = "",color = "Legend") +
    scale_color_manual(values = colors) + theme_bw() + theme(legend.position = 'none')
  
  colors <- c("Stay at Home Survey" = "blue", "Stay at Home Mandate" = "red", "Transit [inverse/4]" = "darkgreen", "R_t"="pink",
              "Cases (thousand)" = "black", "Deaths (ten)" = "brown")
  
  stay_at_home <- joined_data %>% filter(Date>as.Date('2020-10-01')&Date<as.Date('2021-12-31')) %>%
    ggplot(aes(x=Date)) + geom_point(aes(y=stay_at_home,color="Stay at Home Survey")) +
    geom_step(aes(y=C6,color="Stay at Home Mandate")) +
    geom_line(aes(y=-transit_week_mean/25,color="Transit [inverse/4]")) +
    geom_line(aes(y=Cases,color="Cases (thousand)")) +
    geom_line(aes(y=Deaths,color="Deaths (ten)")) +
    ggtitle("Stay at Home") + labs(x = "",y = "",color = "Legend") +
    scale_color_manual(values = colors) + theme_bw() + theme(legend.position = 'none')
  
  colors <- c("Survey" = "blue", "Mandate" = "red", "Transit [inverse/4]" = "darkgreen", "R_t"="pink",
              "Cases (thousand)" = "black", "Deaths (ten)" = "brown")
  
  gatherings_plot <- joined_data %>% filter(Date>as.Date('2020-10-01')&Date<as.Date('2021-12-31')) %>%
    ggplot(aes(x=Date)) + geom_point(aes(y=stay_at_home,color="Survey")) +
    geom_step(aes(y=C4,color="Mandate")) +
    geom_line(aes(y=-transit_week_mean/25,color="Transit [inverse/4]")) +
    geom_line(aes(y=Cases,color="Cases (thousand)")) +
    geom_line(aes(y=Deaths,color="Deaths (ten)")) +
    ggtitle("Restrictions on gatherings") + labs(x = "",y = "",color = "Legend") +
    scale_color_manual(values = colors) + theme_bw()
  
  
  colors <- c("Workplace restrictions Mandate" = "red", "Transit [inverse/4]" = "darkgreen", "R_t"="pink",
              "Cases (thousand)" = "black", "Deaths (ten)" = "brown")
  
  workplaces_plot <- joined_data %>% filter(Date>as.Date('2020-10-01')&Date<as.Date('2021-12-31')) %>%
    ggplot(aes(x=Date)) + 
    geom_step(aes(y=C2,color="Workplace restrictions Mandate")) +
    geom_line(aes(y=-transit_week_mean/25,color="Transit [inverse/4]")) +
    geom_line(aes(y=Cases,color="Cases (thousand)")) +
    geom_line(aes(y=Deaths,color="Deaths (ten)")) +
    ggtitle("Workplace restrictions") + labs(x = "",y = "",color = "Legend") +
    scale_color_manual(values = colors) + theme_bw() + theme(legend.position = 'none')
  
  colors <- c("School restrictions Mandate" = "red", "Transit [inverse/4]" = "darkgreen", "R_t"="pink",
              "Cases (thousand)" = "black", "Deaths (ten)" = "brown")
  
  schools_plot <- joined_data %>% filter(Date>as.Date('2020-10-01')&Date<as.Date('2021-12-31')) %>%
    ggplot(aes(x=Date)) + 
    geom_step(aes(y=C1,color="School restrictions Mandate")) +
    geom_line(aes(y=-transit_week_mean/25,color="Transit [inverse/4]")) +
    geom_line(aes(y=Cases,color="Cases (thousand)")) +
    geom_line(aes(y=Deaths,color="Deaths (ten)")) +
    ggtitle("School restrictions") + labs(x = "",y = "",color = "Legend") +
    scale_color_manual(values = colors) + theme_bw() + theme(legend.position = 'none')
  
  colors <- c("International travel restrictions" = "red", "Transit [inverse/4]" = "darkgreen", "R_t"="pink",
              "Cases (thousand)" = "black", "Deaths (ten)" = "brown")
  
  travel_plot <- joined_data %>% filter(Date>as.Date('2020-10-01')&Date<as.Date('2021-12-31')) %>%
    ggplot(aes(x=Date)) + 
    geom_step(aes(y=C8,color="International travel restrictions")) +
    geom_line(aes(y=-transit_week_mean/25,color="Transit [inverse/4]")) +
    geom_line(aes(y=Cases,color="Cases (thousand)")) +
    geom_line(aes(y=Deaths,color="Deaths (ten)")) +
    ggtitle("International travel restrictions") + labs(x = "",y = "",color = "Legend") +
    scale_color_manual(values = colors) + theme_bw() + theme(legend.position = 'none')
  
  colors <- c("Cancel public events" = "red", "Transit [inverse/4]" = "darkgreen", "R_t"="pink",
              "Cases (thousand)" = "black", "Deaths (ten)" = "brown")
  
  events_plot <- joined_data %>% filter(Date>as.Date('2020-10-01')&Date<as.Date('2021-12-31')) %>%
    ggplot(aes(x=Date)) + 
    geom_step(aes(y=C5,color="Cancel public events")) +
    geom_line(aes(y=-transit_week_mean/25,color="Transit [inverse/4]")) +
    geom_line(aes(y=Cases,color="Cases (thousand)")) +
    geom_line(aes(y=Deaths,color="Deaths (ten)"))  +
    ggtitle("Cancel public events") + labs(x = "",y = "",color = "Legend") +
    scale_color_manual(values = colors) + theme_bw() + theme(legend.position = 'none')
  
  colors <- c("Protection of elderly people" = "red", "Transit [inverse/4]" = "darkgreen", "R_t"="pink",
              "Cases (thousand)" = "black", "Deaths (ten)" = "brown")
  
  elderly_plot <- joined_data %>% filter(Date>as.Date('2020-10-01')&Date<as.Date('2021-12-31')) %>%
    ggplot(aes(x=Date)) + 
    geom_step(aes(y=H8,color="Protection of elderly people")) +
    geom_line(aes(y=-transit_week_mean/25,color="Transit [inverse/4]")) +
    geom_line(aes(y=Cases,color="Cases (thousand)")) +
    geom_line(aes(y=Deaths,color="Deaths (ten)"))  +
    ggtitle("Protection of elderly people") + labs(x = "",y = "",color = "Legend") +
    scale_color_manual(values = colors) + theme_bw() + theme(legend.position = 'none')
  
  NPI_plot <- (workplaces_plot + stay_at_home)/(schools_plot+travel_plot)/(gatherings_plot+elderly_plot)/(mask_plot+events_plot) +
    plot_annotation(tag_levels = 'A') + plot_layout(guides = 'collect')
  
  ggsave('NPI_mandate_plot.pdf',NPI_plot,width=11,height=14)
}

make_NPI_binary <- function(reg_data, threshold = 1)      # 1=max NPI level, 0.66=near max level
{
  reg_data <- reg_data %>% 
    mutate(C1_combined_numeric = as.numeric(C1_combined_numeric >= threshold),
           C2_combined_numeric = as.numeric(C2_combined_numeric >= threshold),
           C3_combined_numeric = as.numeric(C3_combined_numeric >= threshold),
           C4_combined_numeric = as.numeric(C4_combined_numeric >= threshold),
           C6_combined_numeric = as.numeric(C6_combined_numeric >= threshold),
           C8_combined_numeric = as.numeric(C8_combined_numeric >= threshold),
           H6_combined_numeric = as.numeric(H6_combined_numeric >= threshold),
           H8_combined_numeric = as.numeric(H8_combined_numeric >= threshold),
           )
  return(reg_data)
}


get_combined_reg_data <- function(ML=TRUE, i=1, tree_type = 'random_trees_prioritise_settings')
{
  npi_data       <- read_rds('data/NPI_behaviour_data/oxcgrt_npi_data.rds')
  mobility_data  <- read_rds('data/NPI_behaviour_data/google_mobility_cnt.rds') %>% filter(iso3c=='DNK')
  covariant_data <- read_rds('data/NPI_behaviour_data/covariant_org_data.rds') %>% filter(iso3c=='DNK')
  yougov_data    <- read_rds('data/NPI_behaviour_data/yougov_agg_dnk.rds')
  
  #need to lag NPI data to account for lag between infection vs NPI implementation date (as infection occurs before sampling); cite previous 2 NPI papers for this
  npi_data <- npi_data %>% mutate(NPI_Date = dplyr::lag(Date,5))
  
  # Get data
  if(ML)
  {
    node_attrs <- read_csv(paste0('combined_ML_trees/', tree_type, '_combined_ML_trees.csv') ) %>% #, quote="'") %>%
      #dplyr::select(-c('...1')) %>%
      mutate(isoweek = isoweek(SampleDate) + (year(SampleDate)-2020)*52 )
  } else {
    node_attrs <- read_csv(paste0('combined_ML_trees/', tree_type,'_combined_nodelist_',i,'.csv') ) %>% #, quote="'") %>%
      #dplyr::select(-c('...1')) %>%
      mutate(isoweek = isoweek(SampleDate) + (year(SampleDate)-2020)*52 )
  }
 
  household_size_and_sex <- read_csv(paste0('household_sizes_sex_metadata.csv'), quote="'") %>%
    rename(SampleDate = SampleDateTime ) %>%
    mutate(sex_adj = replace_na(SEX,0),                           # if sex is NA encode as 0 for unknown
           hsize   = replace_na(household_size_on_sampledate,1),  # if household size NA assume single person household.
           hsize_small = case_when(hsize<=20 ~ hsize,
                                   TRUE ~ 20 ),            # this only matter for out_degree_address and the max for those households is 6
           hsize_large = case_when(hsize>20 ~ hsize,
                                   TRUE ~ NA ))  
    
  
  serial.interval <- read_csv('data/serial_interval.csv')
  si_short        <- c(serial.interval$fit[1:29],serial.interval$fit[30:100] |> sum())
  rt_iar_file     <- get_partial_sequencing_adjustment()
  
  node_isoweek_dates <- node_attrs %>% group_by(isoweek) %>% summarise(date_min=min(SampleDate),
                                                                       date_max=max(SampleDate),
                                                                       date_mid=mean(c(date_min,date_max)),
                                                                       outdegree_average = mean(out_degree))
  unique_weeks  <- sort(node_attrs$isoweek |> unique())
  
  reg_data <- node_attrs %>% 
    left_join(household_size_and_sex) %>%
    left_join(node_isoweek_dates,by=c('isoweek')) %>%
    left_join(npi_data,by=c('SampleDate'='NPI_Date')) %>%
    left_join(yougov_data,by=c('SampleDate'='Date')) %>%
    left_join(rt_iar_file,by=c('SampleDate'='date')) %>%
    rename(first_vacc_days=days_since_first_vacc,second_vacc_days=days_since_second_vacc) %>%
    mutate(isoweek=as.factor(isoweek),
           vacc_status = as.factor(case_when(first_vacc_complete==FALSE~'unvaccinated',
                                             first_vacc_complete==TRUE&second_vacc_complete==FALSE~'first_vacc',
                                             second_vacc_complete==TRUE~'second_vacc')),
           min_weeks_since_vacc = case_when(is.na(first_vacc_days)&is.na(second_vacc_days) ~ 0,
                                            TRUE ~ floor(pmin( first_vacc_days, second_vacc_days ,na.rm=TRUE)/7)), # add third once we take this into account 
           #min_weeks_since_vacc = case_when(is.na(days_since_first_vacc)&is.na(days_since_second_vacc) ~ 0,
           #                                TRUE ~ floor(pmin( days_since_first_vacc, days_since_second_vacc ,na.rm=TRUE)/7)), # add third once we take this into account 
           weeks_since_vacc = as.factor(case_when(min_weeks_since_vacc==0 ~ '0w',
                                                  min_weeks_since_vacc<3 ~ '<3w',
                                                  min_weeks_since_vacc<10 ~ '3-10w',
                                                  min_weeks_since_vacc<18 ~ '10-18w',
                                                  TRUE ~ '>18w')),
           age_group = as.factor(case_when(Age_at_testing < 11 ~ '0-10yrs',
                                           Age_at_testing < 19 ~ '11-18yrs',
                                           Age_at_testing < 40 ~ '19-39yrs',
                                           Age_at_testing < 60 ~ '40-59yrs',
                                           TRUE ~ '>60yrs')),
           out_degree_family_including_household = replace_na(out_degree_family,0),               # replace NAs as these correspond to the overall out_degree being 0
           out_degree_family                     = replace_na(out_degree_family_ex_household,0),  # replace NAs as these correspond to the overall out_degree being 0
           out_degree_other                      = pmax(out_degree - out_degree_school - out_degree_workplace - out_degree_address,0),
           out_degree_other_ex_family = pmax(out_degree - out_degree_school - out_degree_workplace - out_degree_address - out_degree_family,0))
  
  #school holidays
  school_holidays = list( "Winter 2020"   =c(ymd('2020-02-08'),ymd('2020-02-16')),
                          "Easter 2020"   =c(ymd('2020-04-04'),ymd('2020-04-13')),
                          "Summer 2020"   =c(ymd('2020-06-27'),ymd('2020-08-09')),
                          "Autumn 2020"   =c(ymd('2020-10-10'),ymd('2020-10-18')),
                          "Christmas 2020"=c(ymd('2020-12-19'),ymd('2021-01-03')),
                          "Winter 2021"   =c(ymd('2021-02-13'),ymd('2021-02-21')),
                          "Easter 2021"   =c(ymd('2021-03-27'),ymd('2021-04-05')),
                          "Summer 2021"   =c(ymd('2021-06-26'),ymd('2021-08-09')),
                          "Autumn 2021"   =c(ymd('2021-10-16'),ymd('2021-10-24')),
                          "Christmas 2021"=c(ymd('2021-12-21'),ymd('2022-01-02')),
                          "Winter 2022"   =c(ymd('2022-02-12'),ymd('2022-02-20')),
                          "Easter 2022"   =c(ymd('2022-04-09'),ymd('2022-04-18')),
                          "Summer 2022"   =c(ymd('2022-06-25'),ymd('2022-08-07')),
                          "Autumn 2022"   =c(ymd('2022-10-16'),ymd('2022-10-23')),
                          "Christmas 2022"=c(ymd('2022-12-21'),ymd('2023-01-02')))
  
  school_holiday_vector <- c()
  
  for(hol in school_holidays)
    school_holiday_vector <- c(as.Date(school_holiday_vector),seq(hol[1],hol[2],by='days'))
  
  # shift school holidays by 5 days as the transmission event won't have happened at the testing date but earlier.
  reg_data$is_school_holiday <- as.double(as.Date(reg_data$SampleDate) %in% (school_holiday_vector+5))
  
  reg_data <- reg_data %>% mutate(C1_combined_numeric=1*C1_combined_numeric/3,
                                  C2_combined_numeric=1*C2_combined_numeric/3,
                                  C3_combined_numeric=1*C3_combined_numeric/2,
                                  C4_combined_numeric=1*C4_combined_numeric/4,
                                  C5_combined_numeric=1*C5_combined_numeric/2,
                                  C6_combined_numeric=1*C6_combined_numeric/3,
                                  C7_combined_numeric=1*C7_combined_numeric/2,
                                  C8_combined_numeric=1*C8_combined_numeric/4,
                                  H6_combined_numeric=1*H6_combined_numeric/4,
                                  H8_combined_numeric=1*H8_combined_numeric/2,
                                  avoid_gatherings = (wgt_houseguests + wgt_smallgather + wgt_mediumgather + wgt_largegather + wgt_avoidshops)/5,
                                  stay_at_home     = (wgt_avoidout + wgt_avoidcrowds)/2)
  
  return(reg_data)
}

if(FALSE)
{
  data_ps_ml      <- get_combined_reg_data(ML=TRUE,tree_type = 'random_trees_prioritise_settings') %>% 
    mutate(onward_school_infection  = out_degree_school > 0,
           onward_work_infection    = out_degree_workplace >0, 
           onward_family_infection  = out_degree_family > 0,
           onward_address_infection = out_degree_address >0,
           out_degree_school        = as.integer(out_degree_school),
           out_degree_workplace     = as.integer(out_degree_workplace),
           out_degree_other         = as.integer(out_degree_other),
           school_link              = !is.na(school),
           work_link                = !is.na(workplace),
           Regionskode              = as.factor(Regionskode)
    ) %>%
    filter(SampleDate >= as.Date('2020-09-01'))
}

