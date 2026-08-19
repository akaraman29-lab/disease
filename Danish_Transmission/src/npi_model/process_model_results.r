# Load libraries
library(tidyverse)
library(ggdist)
library(ggh4x)
library(patchwork)
library(meta)
library(latex2exp)

Sys.setlocale("LC_TIME","English_UK") # Need to set as running in Denmark on Danish hardware

source('get_data_functions.R')

get_setting <- function(x)
{
  setting <- rep(NA, length(x))
  for(i in 1:length(x))
  {
    if((str_starts(x[i],'beta_')|str_starts(x[i],'mu_'))&!(str_detect(x[i],'age')|str_detect(x[i],'vacc')|str_detect(x[i],'variant')))
      setting[i] <- str_split(x[i],'_')[[1]][2]
    
    if((str_starts(x[i],'group_intercept')))
      setting[i] <- str_replace(x[i],'group_intercept_','')
  }
  
  return(setting)
}

create_arrow_panel <- function(x)   
{
  arrow_panel_commuity <- ggplot() + annotate('segment',x=0.5,xend=0.5,y=0.5,yend=0,arrow=arrow(length=unit(0.2,'cm')), linewidth=1) +
    xlab('') + ylab('') +
    annotate('text', x=0.52,y=0.45,label= '% of transmission',size=3.5,fontface='italic') + 
    annotate('text', x=0.52,y=0.25,label= x,size=3.5) + 
    annotate('text', x=0.55,y=0.25,label= '',size=3.5) + 
    theme_void() + theme(plot.tag = element_blank())
}

get_NPI <- function(x,y,z, include_C3=TRUE, exclude_H8 = TRUE)
{
  NPI_list <- list()
  
  if(include_C3 & !exclude_H8 )
  {
    NPI_list[['0']] <- 'School restrictions'
    NPI_list[['1']] <- 'Workplace restrictions'
    NPI_list[['2']] <- 'Cancel public events'
    NPI_list[['3']] <- 'Restriction on gathering size'
    NPI_list[['4']] <- 'Stay at home order guidance'
    NPI_list[['5']] <- 'Restrictions on international travel'
    NPI_list[['6']] <- 'Facial coverings'
    NPI_list[['7']] <- 'Protection of elderly people'
  } else if(include_C3 & exclude_H8 ){
    NPI_list[['0']] <- 'School restrictions'
    NPI_list[['1']] <- 'Workplace restrictions'
    NPI_list[['2']] <- 'Cancel public events'
    NPI_list[['3']] <- 'Restriction on gathering size'
    NPI_list[['4']] <- 'Stay at home order guidance'
    NPI_list[['5']] <- 'Restrictions on international travel'
    NPI_list[['6']] <- 'Facial coverings'
  }  else if(!include_C3 & exclude_H8 ){
    NPI_list[['0']] <- 'School restrictions'
    NPI_list[['1']] <- 'Workplace restrictions'
    NPI_list[['2']] <- 'Restriction on gathering size'
    NPI_list[['3']] <- 'Stay at home order guidance'
    NPI_list[['4']] <- 'Restrictions on international travel'
    NPI_list[['5']] <- 'Facial coverings'
  } else {
    NPI_list[['0']] <- 'School restrictions'
    NPI_list[['1']] <- 'Workplace restrictions'
    NPI_list[['2']] <- 'Restriction on gathering size'
    NPI_list[['3']] <- 'Stay at home order guidance'
    NPI_list[['4']] <- 'Restrictions on international travel'
    NPI_list[['5']] <- 'Facial coverings'
    NPI_list[['6']] <- 'Protection of elderly people'
  }
  
  
  NPIs <- rep(NA,length(x))
  
  for(i in 1:length(NPIs))
    if(!is.na(x[i])&!is.na(y[i])&!str_starts(z[i],'group_intercept'))
      NPIs[i] <- NPI_list[[x[i]]]
  
  return(NPIs)
}

get_group <- function(name,id)
{
  age_group <- vacc_status <- variant <- region <- list()
  
  age_group[['0']] <- '>60yrs'
  age_group[['1']] <- '40-59yrs'
  age_group[['2']] <- '19-39yrs'
  age_group[['3']] <- '11-18yrs'
  age_group[['4']] <- '0-10yrs'
  
  vacc_status[['0']] <- 'unvaccinated'
  vacc_status[['1']] <- 'One dose'
  vacc_status[['2']] <- 'Two dose'
  
  variant[['0']] <- 'Wildtype'
  variant[['1']] <- 'Alpha'
  variant[['2']] <- 'Delta'
  variant[['3']] <- 'Eta'
  variant[['4']] <- 'Omicron'
  
  region[['0']] <- 'Nordjydanmark'
  region[['1']] <- 'Midjydanmark'
  region[['2']] <- 'Syddanmark'
  region[['3']] <- 'Hovedstaden'
  region[['4']] <- 'Sjaelland'
  
  group <- rep(NA, length(name))
  
  for(i in 1:length(group))
  {
    if(str_detect(name[i],'age_group'))
      group[i] <- age_group[[id[i]]]
    
    if(str_detect(name[i],'vacc_status'))
      group[i] <- vacc_status[[id[i]]]
    
    if(str_detect(name[i],'variant'))
      group[i] <- variant[[id[i]]]
    
    if(str_detect(name[i],'group_intercept'))
      group[i] <- region[[id[i]]]
  }
  
  return(group)
}

create_coef_table <- function(res_coef, include_C3, exclude_H8, write_file = FALSE)
{
  res_coef <- res_coef %>% filter(!str_starts(coef_name,'mu')) %>%
    separate_wider_delim(coef_name, '[', names = c('coef_name','coef_id'),too_few = "align_start") %>%
    mutate(coef_id = str_replace(coef_id,']',''),
           setting = get_setting(coef_name),
           NPI = get_NPI(coef_id,setting,coef_name, include_C3, exclude_H8 )) %>%
    filter( !(coef_name=='beta_age_group' |                # filter out variables which we have transformed
                coef_name=='beta_vacc_status' |            # use _full instead.
                coef_name=='beta_variant' |  
                str_starts(coef_name,'log_alpha')) ) %>%
    mutate(group = get_group(coef_name,coef_id)) %>%
    mutate(model = str_wrap(model, width = 15)) #%>%
    #filter(!is.na(r_hat))    # filter out reference
  
  #filter out duplicates
  res_coef <- res_coef %>% filter(!str_detect(res_coef$coef_name,'holiday') | 
                                    (str_detect(res_coef$coef_name,'holiday') & str_ends(res_coef$coef_name,'_full')) ) #remove reference group
  res_coef <- res_coef %>% filter(!str_detect(res_coef$coef_name,'holiday') | 
                                    (str_detect(res_coef$coef_name,'holiday') & !is.na(r_hat) )) #remove reference group
  
  
  res_coef['Risk factor'] <- NA
  res_coef[str_detect(res_coef$coef_name,'age_group'),]$`Risk factor` <- 'Age'
  res_coef[str_detect(res_coef$coef_name,'vacc_status'),]$`Risk factor` <- 'Vaccination'
  res_coef[str_detect(res_coef$coef_name,'hurdle'),]$`Risk factor` <- 'Hurdle rate'
  res_coef[str_detect(res_coef$coef_name,'alpha'),]$`Risk factor` <- 'Overdispersion'
  res_coef[str_detect(res_coef$coef_name,'variant'),]$`Risk factor` <- 'Variant'
  res_coef[str_detect(res_coef$coef_name,'holiday'),]$`Risk factor` <- 'Holiday'
  
  res_coef[str_detect(res_coef$coef_name,'hurdle'),]$group <- unlist(lapply(res_coef[str_detect(res_coef$coef_name,'hurdle'),]$coef_name, FUN = function(x){str_split(x,'_')[[1]][2]}))
  res_coef[str_detect(res_coef$coef_name,'alpha'),]$group <- unlist(lapply(res_coef[str_detect(res_coef$coef_name,'alpha'),]$coef_name, FUN = function(x){str_split(x,'_')[[1]][2]}))
  res_coef[str_detect(res_coef$coef_name,'holiday'),]$group <- unlist(lapply(res_coef[str_detect(res_coef$coef_name,'holiday'),]$coef_name, FUN = function(x){str_split(x,'_')[[1]][2]}))
  
  res_coef <- res_coef %>% mutate(setting=case_when(setting=='address'~'Household',
                                                    setting=='other'~'Community',
                                                    setting=='family'~'Family (n/h)',
                                                    TRUE~str_to_title(setting)),
                                  group=case_when(group=='address'~'Household',
                                                  group=='other'~'Community',
                                                  group=='family'~'Family (n/h)',
                                                  TRUE~str_to_title(group)))
  
  res_coef <- res_coef %>% 
    filter(!(str_starts(coef_name,'intercept')|str_starts(coef_name,'sigma'))) %>%
    dplyr::select(NPI,`Risk factor`,setting, group, mean, `hdi_2.5%`, `hdi_97.5%`,mcse_mean,mcse_sd, ess_bulk, ess_tail,r_hat) %>%
    mutate(across(where(is.character), ~replace_na(.x,'-'))) 
  
  if( write_file )
  {
    write_csv(res_coef,'K://SSI_sekvensering_corona/christian/python_scripts/res_coef_ps_ml_zibb_family_noC3_re_ms.csv',
              na = '', col_names = FALSE)  
  } else {
    return(res_coef)
  }
  
}


create_multi_model_plot <- function(res2,plot_type = NA, include_C3 = TRUE, exclude_H8 = FALSE, return_components=FALSE, nrow_in=1 )
{
  res_coef <- res2 %>% filter(!str_starts(coef_name,'mu')) %>%
    separate_wider_delim(coef_name, '[', names = c('coef_name','coef_id'),too_few = "align_start") %>%
    mutate(coef_id = str_replace(coef_id,']',''),
           setting = get_setting(coef_name),
           NPI = get_NPI(coef_id,setting,coef_name, include_C3, exclude_H8 )) %>%
    filter( !(coef_name=='beta_age_group' |                # filter out variables which we have transformed
                coef_name=='beta_vacc_status' |            # use _full instead.
                coef_name=='beta_variant' |  
                str_starts(coef_name,'log_alpha')) ) %>%
    mutate(group = get_group(coef_name,coef_id)) %>%
    mutate(model = str_wrap(model, width = 15)) 
  
  res_coef['Risk factor'] <- NA
  res_coef[str_detect(res_coef$coef_name,'age_group'),]$`Risk factor` <- 'Age'
  res_coef[str_detect(res_coef$coef_name,'vacc_status'),]$`Risk factor` <- 'Vaccination'
  res_coef[str_detect(res_coef$coef_name,'hurdle'),]$`Risk factor` <- 'Hurdle rate'
  res_coef[str_detect(res_coef$coef_name,'alpha'),]$`Risk factor` <- 'Overdispersion'
  res_coef[str_detect(res_coef$coef_name,'variant'),]$`Risk factor` <- 'Variant'
  res_coef[str_detect(res_coef$coef_name,'holiday'),]$`Risk factor` <- 'Holiday'
  
  res_coef[str_detect(res_coef$coef_name,'hurdle'),]$group <- unlist(lapply(res_coef[str_detect(res_coef$coef_name,'hurdle'),]$coef_name, FUN = function(x){str_split(x,'_')[[1]][2]}))
  res_coef[str_detect(res_coef$coef_name,'alpha'),]$group <- unlist(lapply(res_coef[str_detect(res_coef$coef_name,'alpha'),]$coef_name, FUN = function(x){str_split(x,'_')[[1]][2]}))
  res_coef[str_detect(res_coef$coef_name,'holiday'),]$group <- unlist(lapply(res_coef[str_detect(res_coef$coef_name,'holiday'),]$coef_name, FUN = function(x){str_split(x,'_')[[1]][2]}))
  
  res_coef <- res_coef %>% mutate(setting=case_when(setting=='address'~'Household',
                                                    setting=='other'~'Community',
                                                    setting=='family'~'Family (n/h)',
                                                    setting=='tt'~'Total Transmission',
                                                    TRUE~str_to_title(setting)),
                                  group=case_when(group=='address'~'Household',
                                                  group=='other'~'Community',
                                                  group=='family'~'Family (n/h)',
                                                  setting=='tt'~'Total Transmission',
                                                  TRUE~str_to_title(group)))
  
  npi_plt <- res_coef %>% filter(!is.na(NPI)) %>%
    mutate(alpha_size=case_when(`hdi_97.5%`-`hdi_2.5%`>4 ~ 0.8,`hdi_97.5%`-`hdi_2.5%`>2 ~ 0.9, TRUE ~1)) %>% 
    ggplot(aes(y=NPI,x=mean,xmin=`hdi_2.5%`,xmax=`hdi_97.5%`,col=model,group=model),alpha=alpha_size) +
    geom_pointinterval(aes(alpha=alpha_size), position = position_dodge(width=0.2)) + 
    geom_vline(aes(xintercept=0),linetype='dashed') +
    scale_color_lancet() + scale_alpha(guide = 'none') + 
    facet_wrap(~setting,nrow=nrow_in,scales='free_x') + theme_bw() + 
    xlab('Effectiveness &beta;<sub>i</sub> <br> (change in number of secondary infections) <br> for each NPI') + ylab('') +
    theme(axis.title.x = element_markdown(),
      plot.tag = element_text(face='bold'))
  
  design <- matrix(c(3,3,3,3,1,2,2,2),4,2)
  zi_overdispersion_plt <- res_coef %>% filter(is.na(NPI) & `Risk factor` %in% c('Hurdle rate','Overdispersion') ) %>%
    filter(mean<1000) %>%
    mutate(`Risk factor` = case_when(`Risk factor`=='Overdispersion'&group=='Household' ~ 'Overdispersion Beta Binomial',
                                     `Risk factor`=='Overdispersion'&group!='Household' ~ 'Overdispersion Negative Binomial',
                                     TRUE ~ `Risk factor`),
           `Risk factor` = str_replace(`Risk factor`,'Hurdle rate','Zero-inflation probability')) %>%
    group_by(`Risk factor`) %>%
    ggplot(aes(y=group,x=mean,xmin=`hdi_2.5%`,xmax=`hdi_97.5%`,col=model, group=model)) +
    geom_pointinterval(position = position_dodge(width=0.4)) + 
    geom_vline(aes(xintercept=1),linetype='dashed') +
    scale_color_lancet() +
    theme_light() +  ylab('') +
    facet_manual(~`Risk factor`, scales = 'free',design = design) + theme_bw() + xlab('') +
    theme(plot.tag = element_text(face='bold'))
  
  group_order <- c( NA,'Workplace', 'School', 'Household', 'Family (n/h)','Community',
                    "0-10yrs","11-18yrs","19-39yrs","40-59yrs",'>60yrs',
                    "Two Dose","One Dose",'Unvaccinated',
                    "Omicron",'Delta',"Eta","Alpha","Wildtype",
                    "Nordjydanmark", "Midjydanmark", "Syddanmark", "Hovedstaden", "Sjaelland")
  
  risk_factor_data <- res_coef %>% filter(is.na(NPI) & !(`Risk factor` %in% c('Hurdle rate','Overdispersion', NA)) ) %>%
    mutate(mean      = exp(mean),
           `hdi_2.5%`  = exp(`hdi_2.5%`),
           `hdi_97.5%` = exp(`hdi_97.5%`)) %>%
    filter(!(`Risk factor`=='Variant' & mean  > 7) ) %>%  # filter out large value for omicron (as we have too little data for that case)
    filter(!(`Risk factor`=='Holiday' & `hdi_97.5%`  > 2.2 ) ) %>%
    filter(str_ends(coef_name,'_full')) %>%
    filter(!(str_starts(coef_name,'holiday')&coef_id==0)) %>%
    mutate(group=factor(group,levels=group_order))
  
  shaded_data <- risk_factor_data %>%
    filter(abs(mean-1)==0) %>%
    distinct(`Risk factor`, group) 
  
  risk_factor_plt <- ggplot(risk_factor_data,aes(y=group,x=mean,xmin=`hdi_2.5%`,xmax=`hdi_97.5%`)) +
    geom_pointinterval(aes(col=model,group=model),position = position_dodge(width=0.2)) + 
    geom_vline(aes(xintercept=1),linetype='dashed') + 
    scale_color_lancet() +
    theme_light() + theme(legend.position = 'none') + ylab('') +
    facet_wrap(~`Risk factor`, scales = 'free',nrow=1) + theme_bw() + xlab('IRR') +
    theme(plot.tag = element_text(face='bold'))
  
  random_effects_plt <- res_coef %>% filter(str_starts(coef_name,'group_intercept')) %>%
    ggplot(aes(y=group,x=mean,xmin=`hdi_2.5%`,xmax=`hdi_97.5%`,col=model,group=model)) +
    geom_pointinterval(position = position_dodge(width=0.4)) + 
    geom_vline(aes(xintercept=0),linetype='dashed') +
    scale_color_lancet() +
    theme_light() + theme(legend.position = 'none') +
    facet_wrap(~setting, nrow = 1, scales = 'free_x') + ylab('Regional Random Effects') + theme_bw() + xlab('') + ylab('') +
    theme(plot.tag = element_text(face='bold'))# + xlim(c(-1.25,1.25))
  
  if(return_components)
  {
    return(list(npi_plt               = npi_plt,
                risk_factor_plt       = risk_factor_plt,
                random_effects_plt    = random_effects_plt,
                zi_overdispersion_plt = zi_overdispersion_plt))  
  }
  
  if(is.na(plot_type)) {
    plt <- npi_plt / risk_factor_plt / random_effects_plt / zi_overdispersion_plt + 
      plot_annotation(tag_levels = 'A') + plot_layout(guides = 'collect') & theme(legend.position = 'bottom')
    
    return(plt)
  } else if( plot_type == 'reduced_main_text' ){
    plt <- npi_plt / risk_factor_plt + 
      plot_annotation(tag_levels = 'A') + plot_layout(guides = 'collect') & theme(legend.position = 'bottom')
    
    return(plt)
  } 

 
}

# this is plotting the 'simple' Rt vs Rc results.
plot_Rc_setting <- function(model_in='negative_binomial',
                            max_date=as.Date('2021-10-01'),
                            min_date=as.Date('2020-10-01'),
                            sim_data=NULL,Rt_pct=FALSE,
                            plot_type = NA) 
{
  name_ps='Rc_param_estimates_ML_random_trees_prioritise_settings_by_setting_full_period_v3'
  name_rt='Rc_param_estimates_ML_random_trees_by_setting_full_period_v3'
  # plot population level rather an output of model.
  k_results_ps <- read_csv(paste0('data/',name_ps,'.csv')) %>% 
    mutate(setting = case_when(str_ends(random_id,'other_ex_family') ~ 'other_ex_family', 
                               TRUE ~ str_extract(random_id,"[^_]+$"))) %>% unique() %>%
    mutate(setting = case_when(setting=='all'~'Total',
                               setting=='address'~'Household',
                               setting=='other_ex_family'~'Community_ex_family',
                               setting=='other'~'Community',
                               setting=='school'~'School',
                               setting=='workplace'~'Workplace',
                               setting=='family'~'Family (n/h)',
                               setting=='tt'~'Total Transmission',
                               TRUE ~ setting))
  k_results_rt <- read_csv(paste0('data/',name_rt,'.csv')) %>% 
    mutate(setting = case_when(str_ends(random_id,'other_ex_family') ~ 'other_ex_family', 
                               TRUE ~ str_extract(random_id,"[^_]+$"))) %>% unique() %>%
    mutate(setting = case_when(setting=='all'~'Total',
                               setting=='address'~'Household',
                               setting=='other_ex_family'~'Community_ex_family',
                               setting=='other'~'Community',
                               setting=='school'~'School',
                               setting=='workplace'~'Workplace',
                               setting=='family'~'Family (n/h)',
                               setting=='tt'~'Total Transmission',
                               TRUE ~ setting))
  
  if(str_ends(name_ps,'v3'))
  {
    k_results_ps <- k_results_ps %>% filter(setting!='Community') %>% mutate(setting = case_when(setting=='Community_ex_family'~'Community',
                                                                                                 TRUE ~ setting))
    k_results_rt <- k_results_rt %>% filter(setting!='Community') %>% mutate(setting = case_when(setting=='Community_ex_family'~'Community',
                                                                                                 TRUE ~ setting))
  } else {
    k_results_ps <- k_results_ps %>% filter(setting!='Community_ex_family')
    k_results_rt <- k_results_rt %>% filter(setting!='Community_ex_family')
  }
  
  data_rt_ml         <- get_combined_reg_data(ML=TRUE,tree_type = 'random_trees')
  
  if(!exists('node_isoweek_dates'))
    node_isoweek_dates <- data_rt_ml %>% dplyr::select(isoweek,date_min,date_max,date_mid,outdegree_average) %>% unique()
  
  if(!exists('Rc_agg'))
    Rc_agg <- data_rt_ml %>% dplyr::select(Date,Rc,Rc_lb,Rc_ub,partial_sequencing) %>% unique()
  
  # 1 for ML tree
  r_plot_data_ps <- k_results_ps %>% filter(str_detect(random_id,'ML') & isoweek!='all' & model == model_in) %>% 
    filter(parameter!='lp__') %>% 
    dplyr::select(parameter,mean,`2.5%`,`97.5%`,isoweek,setting) %>%
    pivot_wider(names_from = parameter, values_from = c(mean,`2.5%`,`97.5%`),values_fn = mean) %>%
    pivot_longer(
      -c(isoweek,setting),
      names_to = c("quantile", "parameter"),
      names_pattern = "(.*)_(.*)",
      values_to = "value"
    ) %>% 
    filter(parameter %in% c('R')) %>%
    pivot_wider(names_from = quantile,values_from = value) %>%
    left_join(node_isoweek_dates%>% mutate(isoweek=as.character(isoweek)),by=c('isoweek')) %>%
    left_join(Rc_agg,by=c('date_max'='Date')) 
  
  r_plot_data_rt <- k_results_rt %>% filter(str_detect(random_id,'ML') & isoweek!='all' & model == model_in) %>% 
    filter(parameter!='lp__') %>% 
    dplyr::select(parameter,mean,`2.5%`,`97.5%`,isoweek,setting) %>%
    pivot_wider(names_from = parameter, values_from = c(mean,`2.5%`,`97.5%`),values_fn = mean) %>%
    pivot_longer(
      -c(isoweek,setting),
      names_to = c("quantile", "parameter"),
      names_pattern = "(.*)_(.*)",
      values_to = "value"
    ) %>% 
    filter(parameter %in% c('R')) %>%
    pivot_wider(names_from = quantile,values_from = value) %>%
    left_join(node_isoweek_dates%>% mutate(isoweek=as.character(isoweek)),by=c('isoweek')) %>%
    left_join(Rc_agg,by=c('date_max'='Date')) 
  
  k_plot_data_ps <- k_results_ps %>% filter(str_detect(random_id,'ML') & isoweek!='all' & model == model_in) %>% 
    filter(parameter!='lp__') %>% 
    dplyr::select(parameter,mean,`2.5%`,`97.5%`,isoweek,setting) %>%
    pivot_wider(names_from = parameter, values_from = c(mean,`2.5%`,`97.5%`),values_fn = mean) %>%
    pivot_longer(
      -c(isoweek,setting),
      names_to = c("quantile", "parameter"),
      names_pattern = "(.*)_(.*)",
      values_to = "value"
    ) %>% 
    filter(parameter %in% c('k')) %>%
    pivot_wider(names_from = quantile,values_from = value) %>%
    left_join(node_isoweek_dates%>% mutate(isoweek=as.character(isoweek)),by=c('isoweek')) %>%
    left_join(Rc_agg,by=c('date_max'='Date')) 

  k_plot_data_rt <- k_results_rt %>% filter(str_detect(random_id,'ML') & isoweek!='all' & model == model_in) %>% 
    filter(parameter!='lp__') %>% 
    dplyr::select(parameter,mean,`2.5%`,`97.5%`,isoweek,setting) %>%
    pivot_wider(names_from = parameter, values_from = c(mean,`2.5%`,`97.5%`),values_fn = mean) %>%
    pivot_longer(
      -c(isoweek,setting),
      names_to = c("quantile", "parameter"),
      names_pattern = "(.*)_(.*)",
      values_to = "value"
    ) %>% 
    filter(parameter %in% c('k')) %>%
    pivot_wider(names_from = quantile,values_from = value) %>%
    left_join(node_isoweek_dates%>% mutate(isoweek=as.character(isoweek)),by=c('isoweek')) %>%
    left_join(Rc_agg,by=c('date_max'='Date')) 
  
  k_plot_data_rt_address <- k_results_rt %>% filter(str_detect(random_id,'ML') & isoweek!='all' & model == 'beta_binomial_partial_obs') %>% 
    filter(parameter!='lp__') %>% 
    dplyr::select(parameter,mean,`2.5%`,`97.5%`,isoweek,setting) %>%
    pivot_wider(names_from = parameter, values_from = c(mean,`2.5%`,`97.5%`),values_fn = mean) %>%
    pivot_longer(
      -c(isoweek,setting),
      names_to = c("quantile", "parameter"),
      names_pattern = "(.*)_(.*)",
      values_to = "value"
    ) %>% 
    filter(parameter %in% c('k')) %>%
    pivot_wider(names_from = quantile,values_from = value) %>%
    left_join(node_isoweek_dates%>% mutate(isoweek=as.character(isoweek)),by=c('isoweek')) %>%
    left_join(Rc_agg,by=c('date_max'='Date')) 
  
  k_plot_data_ps_address <- k_results_ps %>% filter(str_detect(random_id,'ML') & isoweek!='all' & model == 'beta_binomial_partial_obs') %>% 
    filter(parameter!='lp__') %>% 
    dplyr::select(parameter,mean,`2.5%`,`97.5%`,isoweek,setting) %>%
    pivot_wider(names_from = parameter, values_from = c(mean,`2.5%`,`97.5%`),values_fn = mean) %>%
    pivot_longer(
      -c(isoweek,setting),
      names_to = c("quantile", "parameter"),
      names_pattern = "(.*)_(.*)",
      values_to = "value"
    ) %>% 
    filter(parameter %in% c('k')) %>%
    pivot_wider(names_from = quantile,values_from = value) %>%
    left_join(node_isoweek_dates%>% mutate(isoweek=as.character(isoweek)),by=c('isoweek')) %>%
    left_join(Rc_agg,by=c('date_max'='Date')) 
  
  background_data <- data_rt_ml %>% dplyr::select(date_mid,partial_sequencing) %>% unique() %>%
    filter(date_mid<max_date&date_mid>min_date) %>%
    arrange(date_mid) %>%
    mutate(fill_colour = case_when( partial_sequencing < 0.1 ~ 'Low Sequencing',
                                    partial_sequencing > 0.3 ~ 'High Sequencing',
                                    TRUE ~ 'Medium Sequencing'),
           colour_range = fill_colour != lag(fill_colour, default = first(fill_colour)),
           group = cumsum(colour_range)
    ) %>%
    group_by(group,fill_colour) %>%
    summarise(start=min(date_mid),end=max(date_mid),.groups = 'drop')
  
  r_plot_data_ps_tmp <- r_plot_data_ps %>% group_by(setting) %>% filter(date_max<max_date&date_max>min_date) %>%
    filter(setting=='Total') %>% mutate(parameter=case_when(parameter=='R'~'R_ps', TRUE ~ parameter))
  
  plot_Rt_summary <- r_plot_data_rt %>% group_by(setting) %>% filter(date_max<max_date&date_max>min_date) %>%
    filter(setting=='Total') %>%  mutate(parameter=case_when(parameter=='R'~'R_rt', TRUE ~ parameter)) %>%
    ggplot() +
    geom_rect(data=background_data,aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf, fill=fill_colour),alpha=0.3) +
    geom_line(aes(x=date_mid,y=mean,col=setting,group=setting),lwd=1) + 
    geom_ribbon(aes(x=date_mid,ymin=`2.5%`, ymax=`97.5%`, fill = parameter,  
                    color = parameter), alpha=0.2, linetype = "blank") +
    geom_line(aes(x=date_mid,y=Rc,col='Rc aggregate'), linetype='dashed') +
    geom_ribbon(aes(x=date_mid,ymin=Rc_lb,ymax=Rc_ub,col='Rc aggregate',fill='Rc aggregate'),alpha=0.2, linetype='dashed') +
    geom_point(aes(x=date_mid,y=outdegree_average),shape=8,size=0.7) + 
    geom_line(data=r_plot_data_ps_tmp,
              aes(x=date_mid,y=mean,col=parameter,group=parameter),lwd=1) + 
    geom_ribbon(data=r_plot_data_ps_tmp,
                aes(x=date_mid,ymin=`2.5%`, ymax=`97.5%`, fill = parameter,  
                    color = parameter), alpha=0.2, linetype = "blank") +
    scale_colour_manual("",breaks = c("R",'R_rt','R_ps','Rc aggregate', 'R sim data','Low Sequencing','Medium Sequencing','High Sequencing'), 
                        values=c('red4','red4','lightblue4','lightblue3','darkgreen','darkgrey','lightgrey','white'),
                        labels=c("k",'Random Trees','Prioritised Settings','Rc aggregate', 'R sim data','Low Sequencing','Medium Sequencing','High Sequencing'),
                        guide = "none") +
    scale_fill_manual("",breaks = c("R",'R_rt','R_ps','Rc aggregate', 'R sim data','Low Sequencing','Medium Sequencing','High Sequencing'), 
                      values=c('red4','red4','lightblue4','lightblue3','darkgreen','darkgrey','lightgrey','white'),
                      labels=c("R",'R random trees','R prioritised settings','Rc aggregate', 'R sim data','Low Sequencing','Medium Sequencing','High Sequencing')) +
    coord_cartesian(ylim=c(0.3,2.2)) + theme_minimal() + ylab('') + xlab('') + ggtitle('Reproduction number estimates') + 
    theme(plot.tag = element_text(face='bold'),legend.key = element_rect(fill=NA,colour = 'black')) + 
    guides(fill=guide_legend(title = 'Summary'))
    
  # DO WE WANT TO PLOT NPIs IN THIS 
  k_plot_data_ps_tmp <- k_plot_data_ps %>% group_by(setting) %>% filter(date_max<max_date&date_max>min_date) %>%
    filter(setting=='Total') %>% mutate(parameter=case_when(parameter=='k'~'k_ps', TRUE ~ parameter))
  
  plot_k_summary <- k_plot_data_rt %>% group_by(setting) %>% filter(date_max<max_date&date_max>min_date) %>%
    filter(setting=='Total') %>% mutate(parameter=case_when(parameter=='k'~'k_rt', TRUE ~ parameter)) %>%
    ggplot() +
    geom_rect(data=background_data,aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf, fill=fill_colour),alpha=0.3) +
    geom_line(aes(x=date_mid,y=mean,col=setting,group=setting),lwd=1) + 
    geom_ribbon(aes(x=date_mid,ymin=`2.5%`, ymax=`97.5%`, fill = parameter,  
                    color = parameter), alpha=0.2, linetype = "blank") +
    geom_line(data=k_plot_data_ps_tmp,
              aes(x=date_mid,y=mean,col=parameter,group=parameter),lwd=1) + 
    geom_ribbon(data=k_plot_data_ps_tmp,
                aes(x=date_mid,ymin=`2.5%`, ymax=`97.5%`, fill = parameter,  
                    color = parameter), alpha=0.2, linetype = "blank") +
    scale_colour_manual("",breaks = c("k",'k_rt','k_ps','Rc aggregate', 'R sim data','Low Sequencing','Medium Sequencing','High Sequencing'), 
                        values=c('red4','red4','lightblue4','lightblue3','darkgreen','darkgrey','lightgrey','white'),
                        labels=c("k",'k random trees','k prioritised settings','Rc aggregate', 'R sim data','Low Sequencing','Medium Sequencing','High Sequencing'),
                        guide = 'none') +
    scale_fill_manual("",
                      breaks = c("k",'k_rt','k_ps','Rc aggregate', 'R sim data','Low Sequencing','Medium Sequencing','High Sequencing'), 
                      values=c('red4','red4','lightblue4','lightblue3','darkgreen','darkgrey','lightgrey','white'),
                      labels=c("k",'k random trees','k prioritised settings','Rc aggregate', 'R sim data','Low Sequencing','Medium Sequencing','High Sequencing')) +
    coord_cartesian(ylim=c(0.1,1.4)) + theme_minimal() + ylab('') + xlab('') + ggtitle('Overdispersion Estimates') + 
    theme(plot.tag = element_text(face='bold')) + theme(legend.position="none") + guides(fill="none") 
  
  if(Rt_pct)
  {
    r_plot_data_ps_total <- r_plot_data_ps %>% filter(setting=='Total')
    r_plot_data_ps_pct   <- r_plot_data_ps %>% filter(setting!='Total') %>% 
      left_join(r_plot_data_ps_total,suffix = c('','_Total'),by=c('isoweek','parameter','date_max','date_min','date_mid')) %>%
      mutate(mean   =mean/mean_Total,
             `2.5%` = `2.5%`/`2.5%_Total`,
             `97.5%` = `97.5%`/`97.5%_Total`)
    
    r_plot_data_rt_total <- r_plot_data_rt %>% filter(setting=='Total')
    r_plot_data_rt_pct   <- r_plot_data_rt %>% filter(setting!='Total') %>% 
      left_join(r_plot_data_ps_total,suffix = c('','_Total'),by=c('isoweek','parameter','date_max','date_min','date_mid')) %>%
      mutate(mean   =mean/mean_Total,
             `2.5%` = `2.5%`/`2.5%_Total`,
             `97.5%` = `97.5%`/`97.5%_Total`)
    
    plot_Rc_settings_ps <- r_plot_data_ps_pct %>% group_by(setting) %>% filter(date_max<max_date&date_max>min_date) %>%
      #filter(setting !='all') %>%
      ggplot() +
      geom_rect(data=background_data,aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf, fill=fill_colour),alpha=0.3) +
      geom_line(aes(x=date_mid,y=mean,col=setting,group=setting),lwd=1) + 
      geom_ribbon(aes(x=date_mid,ymin=`2.5%`, ymax=`97.5%`, fill = setting,  
                      color = setting), alpha=0.2, linetype = "blank") +
      scale_colour_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white'),guide = 'none') +
      scale_fill_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white'),
                        limits = c('Total',"Household",'Community', 'School', 'Workplace','Family','Low Sequencing','Medium Sequencing','High Sequencing')) +
      coord_cartesian(ylim=c(0,1)) + theme_minimal() + ylab('') + xlab('') + ggtitle(expression('Proportion of transmission by setting [PS]')) + 
      theme(plot.tag = element_text(face='bold'),legend.key = element_rect(fill=NA,colour = 'black')) + guides(fill=guide_legend(title = 'Settings'))
    
    plot_Rc_settings_rt <- r_plot_data_rt_pct %>% group_by(setting) %>% filter(date_max<max_date&date_max>min_date) %>%
      #filter(setting !='all') %>%
      ggplot() +
      geom_rect(data=background_data,aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf, fill=fill_colour),alpha=0.3) +
      geom_line(aes(x=date_mid,y=mean,col=setting,group=setting),lwd=1) + 
      geom_ribbon(aes(x=date_mid,ymin=`2.5%`, ymax=`97.5%`, fill = setting,  
                      color = setting), alpha=0.2, linetype = "blank") +
      scale_colour_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white'),guide = 'none') +
      scale_fill_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white'),
                        limits = c('Total',"Household",'Community', 'School', 'Workplace','Family','Low Sequencing','Medium Sequencing','High Sequencing')) +
      coord_cartesian(ylim=c(0,1)) + theme_minimal() + ylab('') + xlab('') + ggtitle(expression('Proportion of transmission by setting [RT]')) + 
      theme(plot.tag = element_text(face='bold')) + theme(legend.position = "none") + guides(fill="none")
  } else {
    plot_Rc_settings_ps <- r_plot_data_ps %>% group_by(setting) %>% filter(date_max<max_date&date_max>min_date) %>%
      #filter(setting !='all') %>%
      ggplot() +
      geom_rect(data=background_data,aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf, fill=fill_colour),alpha=0.3) +
      geom_line(aes(x=date_mid,y=mean,col=setting,group=setting),lwd=1) + 
      geom_ribbon(aes(x=date_mid,ymin=`2.5%`, ymax=`97.5%`, fill = setting,  
                      color = setting), alpha=0.2, linetype = "blank") +
      scale_colour_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white'),guide = 'none') +
      scale_fill_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white'),
                        limits = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing')) +
      coord_cartesian(ylim=c(0,2.2)) + theme_minimal() + ylab('') + xlab('') + ggtitle(expression('R'[t]~'estimates by setting [PS]')) + 
      theme(plot.tag = element_text(face='bold'),legend.key = element_rect(fill=NA,colour = 'black')) + guides(fill=guide_legend(title = 'Settings'))
    
    plot_Rc_settings_rt <- r_plot_data_rt %>% group_by(setting) %>% filter(date_max<max_date&date_max>min_date) %>%
      #filter(setting !='all') %>%
      ggplot() +
      geom_rect(data=background_data,aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf, fill=fill_colour),alpha=0.3) +
      geom_line(aes(x=date_mid,y=mean,col=setting,group=setting),lwd=1) + 
      geom_ribbon(aes(x=date_mid,ymin=`2.5%`, ymax=`97.5%`, fill = setting,  
                      color = setting), alpha=0.2, linetype = "dotted") +
      scale_colour_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white')) +
      scale_fill_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white'),
                        limits = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing')) +
      coord_cartesian(ylim=c(0,2.2)) + theme_minimal() + ylab('') + xlab('') + ggtitle(expression('R'[t]~'estimates by setting [RT]')) + 
      theme(plot.tag = element_text(face='bold')) + theme(legend.position = "none") + guides(fill="none")
  }
  
  k_design <- matrix(c(1,2,2,2),4,)
  k_data_ps <- bind_rows(k_plot_data_ps %>% filter(setting != 'Household') %>% mutate(type='Overdispersion Negative Binomial'),
                         k_plot_data_ps_address %>% mutate(type='Overdispersion Beta Binomial') )
  
  plot_k_settings_ps <- k_data_ps %>% group_by(setting) %>% filter(date_max<max_date&date_max>min_date) %>%
    filter(setting !='Total') %>% 
    ggplot() +
    geom_rect(data=background_data,aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf, fill=fill_colour),alpha=0.3) +
    geom_line(aes(x=date_mid,y=mean,col=setting,group=setting),lwd=1) + 
    geom_ribbon(aes(x=date_mid,ymin=`2.5%`, ymax=`97.5%`, fill = setting,  
                    color = setting), alpha=0.2, linetype = "blank") + theme(legend.position = 'none') +
    facet_manual(~type,scales = 'free',design = k_design) +
    scale_colour_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white'),guide = 'none') +
    scale_fill_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white')) +
    theme_minimal() + ylab('') + xlab('') + ggtitle('Overdispersion estimates by setting [PS]') + 
    theme(plot.tag = element_text(face='bold')) + theme(legend.position = "none") + guides(fill="none")
  
  
  k_data_rt <- bind_rows(k_plot_data_rt %>% filter(setting != 'Household') %>% mutate(type='Overdispersion Negative Binomial'),
                        k_plot_data_rt_address %>% mutate(type='Overdispersion Beta Binomial') )
  
  plot_k_settings_rt <- k_data_rt %>% group_by(setting) %>% filter(date_max<max_date&date_max>min_date) %>%
    filter(setting !='Total') %>%
    ggplot() +
    geom_rect(data=background_data,aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf, fill=fill_colour),alpha=0.3) +
    geom_line(aes(x=date_mid,y=mean,col=setting,group=setting),lwd=1) + 
    geom_ribbon(aes(x=date_mid,ymin=`2.5%`, ymax=`97.5%`, fill = setting,  
                    color = setting), alpha=0.2, linetype = "blank") +
    facet_manual(~type,scales = 'free',design = k_design) +
    scale_colour_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white'),guide = 'none') +
    scale_fill_manual("",breaks = c('Total',"Household",'Community', 'School', 'Workplace','Family (n/h)','Low Sequencing','Medium Sequencing','High Sequencing'), values=c('black','red4','blue3','darkgreen','purple4','yellow3','darkgrey','lightgrey','white')) +
    theme_minimal() + ylab('') + xlab('') + ggtitle('Overdispersion estimates by setting [RT]')  + 
    theme(plot.tag = element_text(face='bold')) + theme(legend.position = "none") + guides(fill="none")
  
  #plot for k's for different network construction methods
  design_all <- '
  ab
  ab
  cd
  cd
  cd
  ef
  ef
  ef'
  #plot <- (plot_Rt_summary + plot_k_summary) / (plot_Rc_settings_ps + plot_k_settings_ps) / (plot_Rc_settings_rt + plot_k_settings_rt) + 
  #  plot_annotation(tag_levels = 'A') + plot_layout(guides = 'collect') & theme(legend.position = 'bottom') & 
  #  guides(fill=guide_legend(nrow=6,byrow=TRUE)) 
  
  design_red <- '
  ab
  ab
  cd
  cd
  cd'
  
  if(is.na(plot_type))
  {
    plot <- plot_Rt_summary + plot_k_summary + plot_Rc_settings_ps + plot_k_settings_ps + plot_Rc_settings_rt + plot_k_settings_rt + 
      plot_annotation(tag_levels = 'A') + plot_layout(guides = 'collect',design = design_all) # & theme(legend.position = 'bottom') & 
    #guides(fill=guide_legend(nrow=7,byrow=TRUE)) 
    
    return(plot)  
  } else if( plot_type == 'reduced_main_text' ) {
    plot <- plot_Rt_summary + plot_k_summary + plot_Rc_settings_ps + plot_k_settings_ps + 
      plot_annotation(tag_levels = 'A') + plot_layout(guides = 'collect',design = design_red) & 
      theme(legend.position = 'right') #& 
    #guides(fill=guide_legend(nrow=4,byrow=TRUE)) 
    
    #plot <- plot_Rt_summary + plot_Rc_settings_ps +
    #  plot_annotation(tag_levels = 'A') + plot_layout(guides = 'collect') #& theme(legend.position = 'bottom') 
    
    return(plot) 
  }
    
  
}



#################################################### MAIN TEXT FIGURES (including SI figures for extended version plots)

GENERATE_MAIN_TEXT_FIGURES = FALSE

if(GENERATE_MAIN_TEXT_FIGURES)
{
  models <- list(
    'Prioritised settings ZI-BB'='mcmc_object_ps_ml_zibb_family_noC3_re_reduced_summary.csv',
    'Prioritised settings near max NPI ZI-BB'='mcmc_object_ps_max066_ml_zibb_family_noC3_re_reduced_summary.csv',
    'Prioritised settings seq>30% ZI-BB'='mcmc_object_ps_ml_seq03_zibb_family_noC3_re_reduced_summary.csv'
  )
  
  overall_res <- as_tibble(NULL)
  
  for(model in names(models))
  {
    tmp <- read_csv(models[[model]]) %>% rename(coef_name=`...1`)
    tmp$model <- model
    
    overall_res <- bind_rows(overall_res,tmp)
    
    if(str_detect(models[[model]],'total_transmission'))
      overall_res <- overall_res %>% mutate(coef_name = str_replace(coef_name,"beta_school","beta_tt"))
  }
  
  models_tt <- list(               # tt = Total Transmission
    'Prioritised settings ZI-BB'='mcmc_object_ps_ml_total_transmission_re_reduced_summary.csv',
    'Prioritised settings near max NPI ZI-BB'='mcmc_object_ps_max066_ml_total_transmission_re_reduced_summary.csv',
    'Prioritised settings seq>30% ZI-BB'='mcmc_object_ps_ml_seq03_total_transmission_re_reduced_summary.csv'
  )
  
  overall_tt_res <- as_tibble(NULL)
  
  for(model in names(models_tt))
  {
    tmp <- read_csv(models_tt[[model]]) %>% rename(coef_name=`...1`)
    tmp$model <- model
    
    overall_tt_res <- bind_rows(overall_tt_res,tmp)
    
    if(str_detect(models[[model]],'total_transmission'))
      overall_tt_res <- overall_tt_res %>% mutate(coef_name = str_replace(coef_name,"beta_school","beta_tt"))
  }
  
  
  plt_settings <- create_multi_model_plot(overall_res,plot_type = 'reduced_main_text',include_C3 = FALSE, exclude_H8 = TRUE,return_components = TRUE, nrow_in = 1)
  plt_tt       <- create_multi_model_plot(overall_tt_res,plot_type = 'reduced_main_text',include_C3 = FALSE, exclude_H8 = TRUE,return_components = TRUE, nrow_in = 3)
  
  design_main <- '
   aaaaa
   aaaaa
   aaaaa
   aaaaa
   defgh
   bbbbb
   bbbbb
   bbbbb
   bbbbb
   ccccc
   ccccc
   ccccc
   ccccc'
  
  # percentage of overall transmission
  data_ps_ml_ms <- read_csv('data_ps_ml_ms.csv')
  
  ps_ml_ms_tmp <- c(sum(data_ps_ml_ms$out_degree_family,na.rm = TRUE), sum(data_ps_ml_ms$out_degree_address,na.rm = TRUE), sum(data_ps_ml_ms$out_degree_school,na.rm = TRUE), sum(data_ps_ml_ms$out_degree_workplace,na.rm = TRUE) ) / sum(data_ps_ml_ms$out_degree,na.rm = TRUE) * 100
  ps_ml_ms <- paste0(round(c(100-sum(ps_ml_ms_tmp),ps_ml_ms_tmp),1),'%')
  
  arrow_panel_commuity  <- create_arrow_panel(ps_ml_ms[1])
  arrow_panel_family_nh <- create_arrow_panel(ps_ml_ms[2])
  arrow_panel_household <- create_arrow_panel(ps_ml_ms[3])
  arrow_panel_school    <- create_arrow_panel(ps_ml_ms[4])
  arrow_panel_workplace <- create_arrow_panel(ps_ml_ms[5])
  
  plt <- plt_tt$npi_plt + plt_settings$npi_plt + plt_settings$risk_factor_plt + arrow_panel_commuity + arrow_panel_family_nh + arrow_panel_household + arrow_panel_school + arrow_panel_workplace +      
    plot_annotation(tag_levels = 'A') + plot_layout(guides = 'collect',design = design_main) & theme(legend.position = 'bottom')
  
  ggsave('NPI_plot_incl_family_noC3.pdf',plt,width=13,height=13)
  
  plt <- create_multi_model_plot(overall_res,include_C3 = FALSE, exclude_H8 = TRUE)
  ggsave('NPI_plot_incl_family_noC3_SI.pdf',plt,width=11,height=13)
  
  r_k_plt <- plot_Rc_setting(model='negative_binomial', Rt_pct = FALSE )
  ggsave('r_k_plot.pdf',r_k_plt,width=11,height=14)
  
  r_k_plt <- plot_Rc_setting(model='negative_binomial', Rt_pct = FALSE, plot_type = 'reduced_main_text' ) 
  ggsave('r_k_plot_reduced.pdf',r_k_plt,width=11,height=7) 
  
  coef_table <- create_coef_table(overall_res |> filter(model=='Prioritised settings reduced ZI-BB'),include_C3 = FALSE, exclude_H8 = TRUE, write_file = TRUE)
}

#################################################### SI FIGURES

GENERATE_SI_FIGURES = FALSE

if(GENERATE_SI_FIGURES)
{
  # SI Prioritised Settings vs Random Trees
  models <- list(
    'Prioritised settings ZI-BB'='mcmc_object_ps_ml_zibb_family_noC3_re_reduced_summary.csv',
    'Random trees ZI-BB'='mcmc_object_rt_ml_zibb_family_noC3_re_reduced_summary.csv'
  )
  
  overall_res <- as_tibble(NULL)
  
  for(model in names(models))
  {
    tmp <- read_csv(models[[model]]) %>% rename(coef_name=`...1`)
    tmp$model <- model
     
    overall_res <- bind_rows(overall_res,tmp)
  }
  
  plt <- create_multi_model_plot(overall_res,include_C3 = FALSE)
  ggsave('NPI_plot_SI_PS_vs_RT_noC3.pdf',plt,width=11,height=16)
  
  
  plt <- create_multi_model_plot(overall_res,include_C3 = FALSE, plot_type = 'reduced_main_text')
  ggsave('NPI_plot_SI_PS_vs_RT_sensitivity.pdf',plt,width=11,height=6)
  
  
  ### SI With and without Family
  models <- list(
    'Prioritised setting with family ZI-BB'='mcmc_object_ps_ml_zibb_family_noC3_re_reduced_summary.csv',
    'Prioritised setting ZI-BB'='mcmc_object_ps_ml_zibb_noC3_re_reduced_summary.csv'
  )
  
  overall_res <- as_tibble(NULL)
  
  for(model in names(models))
  {
    tmp <- read_csv(models[[model]]) %>% rename(coef_name=`...1`)
    tmp$model <- model
     
    overall_res <- bind_rows(overall_res,tmp)
  }
  
  plt <- create_multi_model_plot(overall_res)
  ggsave('NPI_plot_SI_with_and_without_family.pdf',plt,width=11,height=16)
  
  plt <- create_multi_model_plot(overall_res,include_C3 = TRUE, plot_type = 'reduced_main_text')
  ggsave('NPI_plot_SI_with_and_without_family_sensitivity.pdf',plt,width=11,height=6)
  
  ### SI different sequencing requirements
  models <- list(
    'Prioritised setting ZI-BB'='mcmc_object_ps_ml_zibb_family_noC3_re_reduced_summary.csv',
    'Prioritised settings seq>10% ZI-BB'='mcmc_object_ps_ml_seq01_zibb_family_noC3_re_reduced_summary.csv',
    'Prioritised settings seq>30% ZI-BB'='mcmc_object_ps_ml_seq03_zibb_family_noC3_re_reduced_summary.csv'
  )
  
  overall_res <- as_tibble(NULL)
  
  for(model in names(models))
  {
    tmp <- read_csv(models[[model]]) %>% rename(coef_name=`...1`)
    tmp$model <- model
    
    overall_res <- bind_rows(overall_res,tmp)
  }
  
  plt <- create_multi_model_plot(overall_res,include_C3 = FALSE)
  ggsave('NPI_plot_SI_seq_noC3.pdf',plt,width=11,height=16)
  
  plt <- create_multi_model_plot(overall_res,include_C3 = FALSE, plot_type = 'reduced_main_text')
  ggsave('NPI_plot_SI_seq_sensitivity.pdf',plt,width=11,height=6)
  
  
  ### SI different random trees + ML
  models <- list(
    'Prioritised settings ZI-BB ML'='mcmc_object_ps_ml_zibb_family_noC3_re_reduced_summary.csv',
    'Prioritised settings ZI-BB 03'='mcmc_object_ps_03_zibb_family_noC3_re_reduced_summary.csv',
    'Prioritised settings ZI-BB 07'='mcmc_object_ps_07_zibb_family_noC3_re_reduced_summary.csv',
    'Prioritised settings ZI-BB 11'='mcmc_object_ps_11_zibb_family_noC3_re_reduced_summary.csv'
  )
  
  overall_res <- as_tibble(NULL)
  
  for(model in names(models))
  {
    tmp <- read_csv(models[[model]]) %>% rename(coef_name=`...1`)
    tmp$model <- model
    
    overall_res <- bind_rows(overall_res,tmp)
  }
  
  plt <- create_multi_model_plot(overall_res,include_C3 = FALSE)
  ggsave('NPI_plot_SI_MLvsIndividual_noC3.pdf',plt,width=11,height=16)
  
  plt <- create_multi_model_plot(overall_res,include_C3 = FALSE, plot_type = 'reduced_main_text')
  ggsave('NPI_plot_SI_MLvsIndividual_sensitivity.pdf',plt,width=11,height=6)
}

