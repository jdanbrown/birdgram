import _ from 'lodash';

import { Place } from 'app/datatypes';

export const HARDCODED_PLACES: Array<Place> = [

  // Auto-generated
  //  - `App.state.ebird.barchartSpecies(props).then(x => print(json(x))).catch(pp)`
  //  - TODO Map ignored species to known species (e.g. like py metadata.ebird.com_names_to_species)

  {
    name: 'CA',
    props: {r: 'US-CA', byr: 2008},
    species: ["BBWD","FUWD","EMGO","SNGO","ROGO","GWFG","TABG","TUBG","BRAN","CACG","CANG","MUSW","TRUS","TUSW","WHOS","EGGO","MUDU","WODU","manduc","GARG","BWTE","CITE","NSHO","GADW","FADU","EUWI","AMWI","NOPI","GWTE","CANV","REDH","COMP","RNDU","TUDU","GRSC","LESC","KIEI","COEI","HADU","SUSC","COSC","BLSC","LTDU","BUFF","COGO","BAGO","HOME","COME","RBME","RUDU","MOUQ","CAQU","GAQU","INPE","CHUK","RNEP","RUGR","GRSG","WTPT","SOGR","WITU","PBGR","HOGR","RNGR","EAGR","WEGR","CLGR","RODO","BTPI","EUCD","AFCD","SPDO","INDO","COGD","RUGD","WWDO","MODO","GBAN","GRRO","YBCU","BBCU","COCU","LENI","CONI","COPO","EWPW","MWPW","BLSW","CHSW","VASW","COSW","WTSW","BTHH","RTHU","BCHU","ANHU","COHU","BTAH","RUHU","ALHU","CAHU","BBIH","VCHU","RIRA","VIRA","SORA","COGA","AMCO","PUGA","YERA","BLRA","SACR","CCRA","BNST","AMAV","AMOY","BLOY","BBPL","AMGP","PAGP","LSAP","SNPL","WIPL","CRPL","SEPL","KILL","MOPL","UPSA","WHIM","LBCU","BTGO","HUGO","MAGO","RUTU","BLTU","REKN","SURF","RUFF","SPTS","STSA","CUSA","RNST","SAND","DUNL","ROSA","PUSA","BASA","LIST","LESA","WRSA","BBSA","PESA","SESA","WESA","SBDO","LBDO","AMWO","COSN","WISN","WIPH","RNPH","REPH","SPSA","SOSA","WATA","GRYE","WILL","LEYE","MASA","WOSA","SPSK","POJA","PAJA","LTJA","COMU","TBMU","PIGU","LBMU","MAMU","SCMU","GUMU","CRMU","ANMU","CAAU","PAAU","RHAU","HOPU","TUPU","STGU","BLKI","IVGU","SAGU","BOGU","BHGU","LIGU","ROGU","LAGU","FRGU","BTGU","HEEG","MEGU","RBGU","WEGU","YFGU","CAGU","HERG","ICGU","LBBG","SBGU","GWGU","GLGU","GBBG","KEGU","SOTE","BRTE","LETE","GBTE","CATE","BLTE","COTE","ARTE","FOTE","ROYT","ELTE","BLSK","RBTR","RTTR","RTLO","ARLO","PALO","COLO","YBLO","SAAL","LAAL","BFAL","STAL","WISP","FTSP","LESP","TOSP","ASSP","BSTP","WRSP","BLSP","LSTP","NOFU","GFPE","KEPE","MUPE","MOPE","HAPE","COPE","STPE","JOPE","WCPE","COSH","PFSH","FFSH","GRSH","WTSH","BULS","SOSH","SRTS","MASH","BVSH","WOST","MAFR","GREF","MABO","NABO","BFBO","BRBO","RFBO","NOGA","BRAC","PECO","NECO","DCCO","AWPE","BRPE","AMBI","LEBI","GBHE","GREG","SNEG","LBHE","TRHE","REEG","CAEG","GRHE","BCNH","YCNH","WHIB","GLIB","WFIB","ROSP","CACO","BLVU","TUVU","OSPR","WTKI","STKI","GOEA","MIKI","NOHA","SSHA","COHA","NOGO","BAEA","COBH","HASH","GRHA","RSHA","BWHA","SWHA","ZTHA","RTHA","RLHA","FEHA","BANO","FLOW","WESO","GHOW","SNOW","NOPO","ELOW","BUOW","SPOW","BADO","GGOW","LEOW","SEOW","NSWO","BEKI","EUWR","WISA","YBSA","RNSA","RBSA","LEWO","ACWO","GIWO","BBWO","DOWO","NUWO","LBWO","HAWO","WHWO","PIWO","NOFL","GIFL","CRCA","AMKE","MERL","GYRF","PEFA","PRFA","RRPA","MOPA","WWPA","yecpar","RCPA","LCPA","RLPA","YHPA","YCPA","TFAM","WFPA","NAPA","bucpar","MIPA","RMPA","OSFL","GRPE","WEWP","EAWP","YBFL","ALFL","WIFL","LEFL","HAFL","GRFL","DUFL","PSFL","COFL","BBFL","BLPH","EAPH","SAPH","VEFL","DCFL","ATFL","GCFL","BCFL","SBFL","TRKI","COKI","CAKI","TBKI","WEKI","EAKI","STFL","BROS","LOSH","NSHR","WEVI","BEVI","GRVI","HUVI","YTVI","CAVI","BHVI","PLVI","PHVI","WAVI","YGVI","GRAJ","BTMJ","PIJA","STJA","BLJA","ISSJ","CASJ","WOSJ","BBMA","YBMA","CLNU","AMCR","CORA","HOLA","EUSK","NRWS","PUMA","TRES","VGSW","BANS","BARS","CLSW","CASW","BCCH","MOCH","CBCH","OATI","JUTI","VERD","BUSH","RBNU","WBNU","PYNU","BRCR","ROWR","CANW","HOWR","PAWR","WIWR","SEWR","MAWR","BEWR","CACW","BGGN","CAGN","BTGN","AMDI","RWBU","GCKI","RCKI","DUWA","ARWA","WREN","JAWE","BLUE","RFBL","NOWH","WEBL","MOBL","TOSO","VATH","VEER","GCTH","SWTH","HETH","WOTH","AMRO","RBRO","GRCA","CBTH","BRTH","BETH","CATH","LCTH","CRTH","SATH","NOMO","EUST","EYWA","CIWA","WHWA","OBPI","RTPI","AMPI","SPPI","BOWA","CEDW","PHAI","BRAM","EVGR","HAWF","PIGR","GCRF","BLRF","HOFI","PUFI","CAFI","CORE","RECR","WWCR","EUGO","PISI","LEGO","LAGO","AMGO","LALO","CCLO","SMLO","MCLO","SNBU","LIBU","RUBU","CASP","GRSP","CHSP","CCSP","BCSP","FISP","BRSP","BTSP","LASP","LARB","ATSP","FOSP","DEJU","WCSP","GCSP","HASP","WTSP","SABS","BESP","VESP","LCSP","NESP","SAVS","BAIS","SOSP","LISP","SWSP","ABTO","CALT","RCSP","GTTO","SPTO","YBCH","YHBL","BOBO","WEME","EAME","OROR","HOOR","SBAO","BUOR","BAOR","SCOR","RWBL","TRBL","BROC","BHCO","RUBL","BRBL","COGR","GTGR","OVEN","WEWA","LOWA","NOWA","GWWA","BWWA","BAWW","PROW","TEWA","OCWA","LUWA","NAWA","VIWA","CONW","MGWA","MOWA","KEWA","COYE","HOWA","AMRE","CMWA","CERW","NOPA","TRPA","MAWA","BBWA","BLBW","YEWA","CSWA","BLPW","BTBW","PAWA","PIWA","YRWA","YTWA","PRAW","GRWA","BTYW","TOWA","HEWA","BTNW","CAWA","WIWA","RFWA","PARE","HETA","SUTA","SCTA","WETA","NOCA","PYRR","RBGR","BHGR","BLGR","LAZB","INBU","VABU","PABU","DICK","HOSP","NRBI","OCHW","BRMA","SBMU","PTWH"],
  },

  {
    name: 'TX',
    props: {r: 'US-TX', byr: 2008},
    species: ["BBWD","FUWD","SNGO","ROGO","GRGO","swagoo1","GWFG","BRAN","CACG","CANG","MUSW","TRUS","TUSW","EGGO","MUDU","RITE","WODU","BWTE","CITE","NSHO","GADW","EUWI","AMWI","ABDU","MODU","WCHP","NOPI","GWTE","CANV","REDH","RNDU","GRSC","LESC","KIEI","SUSC","BLSC","LTDU","BUFF","COGO","BAGO","HOME","COME","RBME","MADU","RUDU","PLCH","HELG","NOBO","SCQU","GAQU","MONQ","INPE","RNEP","GRPC","LEPC","WITU","AMFL","grefla3","LEGR","PBGR","HOGR","RNGR","EAGR","WEGR","CLGR","RODO","WCPI","RBPI","BTPI","EUCD","AFCD","INDO","COGD","RUGD","WTDO","WWDO","MODO","GBAN","GRRO","YBCU","MACU","BBCU","LENI","CONI","COPA","COPO","CWWI","EWPW","MWPW","CHSW","WTSW","MEVI","GNBM","RIHU","ATHU","BTHH","LUHU","RTHU","BCHU","ANHU","COHU","BTAH","RUHU","ALHU","CAHU","BBIH","BBEH","VCHU","WEHU","KIRA","CLRA","VIRA","SORA","COGA","AMCO","PUGA","YERA","BLRA","SACR","CCRA","WHCR","BNST","AMAV","AMOY","BBPL","AMGP","COPL","SNPL","WIPL","SEPL","PIPL","KILL","MOPL","NOJA","UPSA","WHIM","LBCU","BTGO","BTGD","HUGO","MAGO","RUTU","REKN","SURF","RUFF","STSA","CUSA","RNST","SAND","DUNL","PUSA","BASA","LESA","WRSA","BBSA","PESA","SESA","WESA","SBDO","LBDO","AMWO","WISN","WIPH","RNPH","REPH","SPSA","SOSA","GRYE","WILL","LEYE","SPSK","POJA","PAJA","LTJA","BLKI","SAGU","BOGU","LIGU","LAGU","FRGU","HEEG","MEGU","RBGU","WEGU","CAGU","HERG","ICGU","LBBG","SBGU","GLGU","GBBG","KEGU","BRNO","SOTE","BRTE","LETE","GBTE","CATE","BLTE","COTE","ARTE","FOTE","ROYT","SATE","ELTE","BLSK","WTTR","RBTR","RTLO","PALO","COLO","LESP","BSTP","COSH","GRSH","SOSH","MASH","AUSH","JABI","WOST","MAFR","MABO","BRBO","NOGA","ANHI","NECO","DCCO","AWPE","BRPE","AMBI","LEBI","BTTH","GBHE","GREG","SNEG","LBHE","TRHE","REEG","CAEG","GRHE","BCNH","YCNH","WHIB","GLIB","WFIB","ROSP","BLVU","TUVU","OSPR","WTKI","HBKI","STKI","GOEA","SNKI","DTKI","MIKI","NOHA","SSHA","COHA","NOGO","BAEA","COBH","GBLH","ROHA","HASH","WTHA","GRHA","RSHA","BWHA","STHA","SWHA","ZTHA","RTHA","RLHA","FEHA","BANO","FLOW","WESO","EASO","GHOW","SNOW","NOPO","FEPO","ELOW","BUOW","SPOW","BADO","LEOW","SEOW","NSWO","ELTR","RIKI","BEKI","AMKI","GKIN","WISA","YBSA","RNSA","LEWO","RHWO","ACWO","GFWO","RBWO","DOWO","LBWO","RCWO","HAWO","PIWO","NOFL","CRCA","AMKE","MERL","APFA","PEFA","PRFA","succoc","MOPA","WWPA","RCPA","LCPA","RLPA","YHPA","WFPA","NAPA","BAYM","CFMA","GREP","NOBT","WCEL","TUFL","OSFL","GRPE","WEWP","EAWP","YBFL","ACFL","ALFL","WIFL","LEFL","HAFL","GRFL","DUFL","PSFL","COFL","BBFL","BLPH","EAPH","SAPH","VEFL","DCFL","ATFL","NUFL","GCFL","BCFL","GKIS","SBFL","PIRF","VAFL","TRKI","COKI","CAKI","WEKI","EAKI","GRAK","STFL","FTFL","RTBE","LOSH","NSHR","BCVI","WEVI","BEVI","GRVI","HUVI","YTVI","CAVI","BHVI","PLVI","PHVI","WAVI","YGVI","BWVI","BRJA","GREJ","PIJA","STJA","BLJA","WOSJ","MEJA","BBMA","CLNU","AMCR","TACR","FICR","CHRA","CORA","HOLA","NRWS","PUMA","TRES","VGSW","BANS","BARS","CLSW","CASW","CACH","MOCH","JUTI","TUTI","BCTI","VERD","BUSH","RBNU","WBNU","PYNU","BHNU","BRCR","ROWR","CANW","HOWR","WIWR","SEWR","MAWR","CARW","BEWR","CACW","BGGN","BTGN","AMDI","RVBU","GCKI","RCKI","NOWH","EABL","WEBL","MOBL","TOSO","VATH","VEER","GCTH","SWTH","HETH","WOTH","AZTH","WTTH","CCTH","AMRO","RBRO","GRCA","CBTH","BRTH","LBTH","CRTH","SATH","TRMO","NOMO","EUST","AMPI","SPPI","CEDW","PHAI","EVGR","GCRF","HOFI","PUFI","CAFI","CORE","RECR","WWCR","EUGO","PISI","LEGO","LAGO","AMGO","LALO","CCLO","SMLO","MCLO","BOSP","CASP","BACS","GRSP","OLSP","CHSP","CCSP","BCSP","FISP","BRSP","BTSP","LASP","LARB","ATSP","FOSP","DEJU","YEJU","WCSP","GCSP","HASP","WTSP","SABS","SSPA","VESP","LCSP","SESP","NESP","SAVS","BAIS","HESP","SOSP","LISP","SWSP","CANT","RCSP","GTTO","SPTO","EATO","YBCH","YHBL","BOBO","WEME","EAME","BVOR","OROR","HOOR","BUOR","ALOR","AUOR","BAOR","SCOR","RWBL","BROC","BHCO","RUBL","BRBL","COGR","BTGR","GTGR","OVEN","WEWA","LOWA","NOWA","GWWA","BWWA","BAWW","PROW","SWWA","TEWA","OCWA","COLW","LUWA","NAWA","VIWA","CONW","GCYE","MGWA","MOWA","KEWA","COYE","HOWA","AMRE","CMWA","CERW","NOPA","TRPA","MAWA","BBWA","BLBW","YEWA","CSWA","BLPW","BTBW","PAWA","PIWA","YRWA","YTWA","PRAW","GRWA","BTYW","TOWA","HEWA","GCWA","BTNW","RCWA","GCRW","CAWA","WIWA","RFWA","PARE","STRE","HETA","SUTA","SCTA","WETA","FCTA","CCGR","NOCA","PYRR","RBGR","BHGR","BLBU","BLGR","LAZB","INBU","VABU","PABU","DICK","RLHO","SAFI","YFGR","HOSP","NRBI","OCHW","BRMA","SBMU"],
  },

  {
    name: 'Costa Rica',
    props: {r: 'CR', bmo: 12, emo: 2, byr: 2008},
    species: ["HITI","GRTI","LITI","SBTI","THTI","BBWD","FUWD","MUDU","BWTE","CITE","NSHO","AMWI","NOPI","GWTE","CANV","REDH","RNDU","LESC","HOME","MADU","RUDU","PLCH","GHEC","CRGU","BLAG","GRCU","TFQU","BCWP","CRBO","MAWQ","BEWQ","BBWQ","SPWQ","LEGR","PBGR","RODO","PVPI","SCPI","WCPI","RBPI","BTPI","RUDP","SBPI","EUCD","INDO","COGD","PBGD","RUGD","BLGD","MCGD","RUQD","VIQD","OBQD","WTDO","GCDO","GHDO","BFQD","PBQD","CHQD","WWDO","MODO","GRTA","SBAN","GBAN","STCU","PHCU","LEGC","RVGC","SQCU","YBCU","MACU","COCC","BBCU","LENI","CONI","SHTN","COPA","WTNI","CWWI","RUNI","EWPW","DUNI","GRPO","CPOT","NORP","BLSW","WCHS","SFSW","CCSW","WCSW","CHSW","VASW","CRSW","GRSW","LSTS","WNJA","WTSI","BRHE","BTBA","GREH","LBIH","STHR","GFRL","BRVI","LEVI","PCFA","GNBM","VEMA","GRET","BCCO","WCCO","GCBR","TAHU","LBST","PCST","FTHU","WBMG","PTMG","WTMG","MTWO","RTHU","VOHU","SCHU","CAEM","GAEM","VHHU","SBRH","VISA","BTPL","CRWO","STHM","BLBH","WTEM","CHEM","SNOC","BCHH","CHHU","MAHU","SVHU","SBEH","RTAH","CIHU","SHTH","BTRG","MARA","PBCR","SPRA","UNIC","RNWR","RSWR","GCWR","SORA","COGA","AMCO","PUGA","OCCR","YBCR","WTCR","GBCR","SUNG","LIMP","DSTK","BNST","AMAV","AMOY","BBPL","AMGP","SOLA","COPL","SNPL","WIPL","SEPL","KILL","NOJA","WAJA","WHIM","LBCU","MAGO","RUTU","REKN","SURF","STSA","SAND","DUNL","BASA","LESA","WRSA","PESA","SESA","WESA","SBDO","LBDO","WISN","WIPH","RNPH","REPH","SPSA","SOSA","WATA","GRYE","WILL","LEYE","POJA","PAJA","LTJA","BLKI","SAGU","BOGU","LAGU","FRGU","RBGU","HERG","GBBG","KEGU","BRNO","WHTT","SOTE","BRTE","LETE","GBTE","CATE","BLTE","COTE","FOTE","ROYT","SATE","ELTE","BLSK","SUNB","RBTR","WISP","LESP","WRSP","BLSP","LSTP","PFSH","WTSH","GASH","JABI","WOST","MAFR","GREF","MABO","NABO","BFBO","BRBO","RFBO","ANHI","NECO","AWPE","BRPE","PIBI","LEBI","RTHE","FTHE","BTTH","GBHE","GREG","SNEG","LBHE","TRHE","REEG","CAEG","GRHE","STRH","AGHE","WHHE","BCNH","YCNH","BBHE","WHIB","GLIB","GRIB","ROSP","KIVU","BLVU","TUVU","LYHV","OSPR","PEKI","WTKI","HBKI","GHKI","STKI","CREA","HAEA","BLHE","ORHE","BAWH","BCHA","SNKI","DTKI","MIKI","PLKI","NOHA","GBEH","TIHA","SSHA","COHA","BIHA","CRHA","COBH","SAHA","GBLH","SOEA","BAHA","ROHA","HASH","WTHA","WHHA","SEHA","GRHA","GLHA","BWHA","STHA","SWHA","ZTHA","RTHA","BANO","BSSO","TRSO","VESO","PASO","CROW","SPEO","GHOW","CRPO","CAPO","FEPO","MOOW","BLWO","STRO","USWO","REQU","LTTR","STTR","BHTR","BATR","GATR","BTHT","ELTR","OBTR","COTR","TOMO","LEMO","RMOT","KBMO","BBMO","TBMO","RIKI","BEKI","AMKI","APKI","GKIN","GARK","WNPU","PIPU","WWPU","LAMO","WFNU","RTJA","GJAC","RHBA","PBBA","NOET","COAR","FBAR","YETO","YTTO","KBTO","OLPI","YBSA","ACWO","GNWO","BCWO","RCRW","HOWO","HAWO","SMBW","RRWO","PBIW","LIWO","CIWO","CCOW","RWWO","GOWO","BAFF","SBFF","COFF","RTCA","CRCA","YHCA","LAFA","AMKE","MERL","APFA","BAFA","PEFA","RFPA","BAPA","OCPA","BHOP","BHEP","WCPA","RLPA","YNPA","WFPA","MEAP","SWPA","OTPA","OFPA","BTPA","GGMA","SCMA","CFPA","RRAN","FAAN","GANT","BAAN","BCAS","BHOA","RUAN","PLAN","STCA","SPCA","CTAN","WFLA","SLAN","DWAN","DUAN","BACA","CBAN","DMAN","ZEAN","BIAN","SPAN","OCAN","BCAP","SCAA","SCHA","THAN","OBAN","SFTA","BFAN","BHEA","RBAN","TTLE","STLE","GTLE","OLWO","LTWO","RUWO","TWWO","PBRW","WBWO","NOBW","BBNW","SNBW","COWO","IBIW","BSWO","SPWO","BBSC","SHWO","SCRW","PLXE","STXE","BUTU","BFFG","STFG","LIFG","RUFG","SBTR","BTFG","STPW","SPBA","RUTR","RFSP","SLSP","PBSP","YBTY","BCTY","NOBT","SOBT","MCTY","COCF","YETY","YCTY","GREL","YBEL","LEEL","MOEL","TOTY","OSTF","OBFL","SECF","SLCF","RBTY","RLTY","NOSF","BPYT","SCPT","NOBE","SHTF","COTF","BHTF","ERFL","YOFL","YMFL","STTS","WTRS","GCRS","ROFL","RDTF","SRFL","BTFL","BCOF","TCFL","TUFL","OSFL","DAPE","OCPE","WEWP","EAWP","TROP","YBFL","ACFL","ALFL","WIFL","WTFL","LEFL","YEFL","BCAF","BLPH","LTTY","BRAT","RMOU","DCFL","PAFL","NUFL","GCFL","BCFL","GKIS","BOBF","RMFL","SOFL","GCAF","WRFL","GBFL","STRF","SBFL","PIRF","TRKI","WEKI","EAKI","GRAK","STFL","FTFL","SHAR","PTFR","BNUM","LOCO","TUCO","RUFP","TWBE","YBCO","SNCO","LATM","LOTM","WRMA","BCRM","WCOM","OCMA","WCRM","RCMA","GHPI","BCRT","MATI","NOSC","SPMO","BABE","CIMB","WWBE","BAWB","RTBE","RBPE","SCRG","GRSV","TCGR","LESG","WEVI","MAVI","YTVI","YWVI","BHVI","PHVI","WAVI","BCAV","YGVI","STHJ","AHJA","WTMJ","BRJA","BCHJ","BAWS","NRWS","SRWS","PUMA","GYBM","BCMA","TRES","MANS","VGSW","BANS","BARS","CLSW","CASW","NIWR","SCBW","HOWR","OCWR","TIWR","SEWR","BABW","RNAW","BBEW","RBSW","SBSW","BTWR","BANW","RAWW","SIBW","CABW","CAKW","ISWR","RIWR","BAYW","WBWW","GBWW","SONW","TFGN","LBGN","WLGN","TRGN","AMDI","BFSO","BBNT","OBNT","SBNT","RCNT","BHNT","VEER","GCTH","SWTH","WOTH","MOTH","PVTH","WTTH","CCTH","SOOT","GRCA","TRMO","AMPI","CEDW","BAYS","LTSF","GBCH","SEUP","YCEU","TBEU","YTEU","ELEU","SPCE","OBAE","WVEU","TCEU","LEGO","YBSI","ROTT","ATCH","SCCH","COCL","SHSP","BOSP","GRSP","OLSP","BSTS","CCSP","CRBR","OBSP","CCBR","SFFI","VOJU","RCOS","WCSP","SAVS","LISP","LFFI","WEGS","CAGS","RUSP","YTFI","WNBR","WRET","YBCH","EAME","RBBL","YBIC","CROR","CHOR","MORO","SRCA","BCOR","OROR","YBOR","YTOR","SBAO","BUOR","SBOR","BAOR","RWBL","SHCO","BROC","GICO","MEBL","GTGR","NIGR","OVEN","WEWA","LOWA","NOWA","GWWA","BWWA","BAWW","PROW","FTHW","TEWA","OCWA","NAWA","GCYE","MGWA","MOWA","KEWA","OCYE","COYE","HOWA","AMRE","CMWA","CERW","NOPA","TRPA","MAWA","BBWA","BLBW","YEWA","CSWA","BLPW","BTBW","PAWA","YRWA","YTWA","TOWA","HEWA","GCWA","BTNW","RCWA","BCWA","GCRW","CRWA","BURW","CAWA","WIWA","STRE","COLR","DFTA","HETA","SUTA","SCTA","WETA","FCTA","WWTA","RCAT","RTAT","BCAT","CATA","BFAG","BTGG","RBGR","BLSE","BLGR","INBU","PABU","DICK","GHET","WSTA","TCTA","WLTA","WTST","CCTA","FRTA","BAGT","BGTA","YWTA","PALT","SPTA","GHOT","SCHT","PCTA","RWTA","BHTA","EMTA","STTA","STDA","BLDA","SHHO","RLHO","GRHO","SRTA","BAYT","SLFL","SLFI","PBFI","WTGF","BGRA","RBSE","TBSF","NISF","VASE","YBSE","SCSE","BANA","YFGR","COFI","BTSA","BHSA","GRAS","SSAL","SCOG","HOSP","TRMU"],
  },

  {
    name: 'Rancho Naturalista',
    props: {r: 'L468901', bmo: 12, emo: 2, byr: 2008},
    species: ["HITI","GRTI","LITI","BBWD","MUDU","GHEC","CRGU","BLAG","RODO","PVPI","RBPI","BTPI","RUDP","SBPI","INDO","RUGD","RUQD","WTDO","GCDO","PBQD","CHQD","WWDO","GBAN","STCU","SQCU","SHTN","COPA","CWWI","CPOT","CCSW","WCSW","VASW","GRSW","WNJA","WTSI","BTBA","GREH","LBIH","STHR","GFRL","BRVI","LEVI","PCFA","GNBM","GRET","BCCO","GCBR","TAHU","WBMG","PTMG","WTMG","RTHU","SCHU","GAEM","VHHU","SBRH","VISA","BTPL","CRWO","BLBH","CHEM","SNOC","SVHU","RTAH","RSWR","PUGA","WTCR","SOLA","NOJA","SPSA","SUNB","ANHI","NECO","FTHE","GBHE","GREG","SNEG","LBHE","CAEG","GRHE","BCNH","BBHE","GRIB","KIVU","BLVU","TUVU","OSPR","WTKI","HBKI","GHKI","STKI","BLHE","DTKI","SSHA","COHA","BIHA","GBLH","BAHA","ROHA","WHHA","GRHA","BWHA","STHA","ZTHA","RTHA","CROW","SPEO","CRPO","FEPO","MOOW","STRO","STTR","GATR","BTHT","COTR","LEMO","RMOT","BBMO","RIKI","AMKI","GKIN","LAMO","RTJA","RHBA","PBBA","NOET","COAR","YTTO","KBTO","YBSA","ACWO","BCWO","HOWO","SMBW","PBIW","LIWO","RWWO","GOWO","BAFF","CRCA","YHCA","LAFA","MERL","BAFA","PEFA","RFPA","BAPA","OCPA","BHOP","WCPA","SWPA","OTPA","CFPA","RRAN","FAAN","BAAN","RUAN","PLAN","CTAN","SLAN","DWAN","DUAN","DMAN","ZEAN","BIAN","SPAN","THAN","SFTA","BHEA","RBAN","TTLE","STLE","GTLE","OLWO","RUWO","PBRW","WBWO","NOBW","COWO","SPWO","BBSC","SHWO","SCRW","PLXE","STXE","BFFG","LIFG","BTFG","SPBA","RUTR","RFSP","SLSP","YETY","GREL","YBEL","LEEL","MOEL","TOTY","OSTF","OBFL","SLCF","RBTY","RLTY","SCPT","NOBE","COTF","BHTF","ERFL","YOFL","YMFL","WTRS","RDTF","SRFL","TCFL","TUFL","OSFL","DAPE","WEWP","EAWP","TROP","YBFL","ACFL","WIFL","WTFL","LEFL","YEFL","BLPH","LTTY","BRAT","RMOU","DCFL","GCFL","GKIS","BOBF","SOFL","GCAF","WRFL","GBFL","SBFL","PIRF","TRKI","RUFP","WRMA","WCOM","WCRM","RCMA","GHPI","BCRT","MATI","NOSC","BABE","CIMB","WWBE","RBPE","TCGR","LESG","YTVI","PHVI","BCAV","YGVI","BRJA","BAWS","NRWS","SRWS","PUMA","GYBM","BANS","BARS","NIWR","SCBW","HOWR","OCWR","BABW","BTWR","SIBW","CABW","BAYW","WBWW","GBWW","SONW","TFGN","LBGN","TRGN","AMDI","BFSO","OBNT","SBNT","BHNT","SWTH","WOTH","MOTH","PVTH","WTTH","CCTH","GRCA","TRMO","CEDW","BAYS","LTSF","GBCH","YCEU","YTEU","ELEU","OBAE","WVEU","TCEU","LEGO","ATCH","COCL","BSTS","OBSP","CCBR","SFFI","RCOS","WEGS","YTFI","WNBR","YBCH","EAME","RBBL","YBIC","CHOR","MORO","SRCA","BCOR","OROR","BAOR","SHCO","BROC","GICO","MEBL","GTGR","OVEN","WEWA","LOWA","NOWA","GWWA","BWWA","BAWW","TEWA","GCYE","MGWA","MOWA","KEWA","OCYE","COYE","AMRE","TRPA","MAWA","BBWA","BLBW","YEWA","CSWA","PAWA","YRWA","YTWA","TOWA","BTNW","RCWA","GCRW","CRWA","BURW","CAWA","WIWA","STRE","COLR","HETA","SUTA","WWTA","RTAT","CATA","BFAG","BTGG","RBGR","INBU","WSTA","TCTA","WLTA","CCTA","BAGT","BGTA","PALT","SPTA","GHOT","SCHT","PCTA","BHTA","EMTA","STTA","STDA","BLDA","SHHO","RLHO","GRHO","BAYT","SLFL","BGRA","TBSF","VASE","YBSE","BANA","YFGR","BTSA","BHSA","GRAS","HOSP"],
  },

  {
    name: 'Rancho Naturalista--Cerro El Silencio',
    props: {r: 'L1127254', bmo: 12, emo: 2, byr: 2008},
    species: ["GHEC","BLAG","BBWQ","RBPI","BTPI","RUDP","SBPI","RUGD","WTDO","WWDO","GBAN","SQCU","COPA","CPOT","WCSW","VASW","WNJA","BTBA","GREH","STHR","GFRL","BRVI","PCFA","GNBM","GRET","GCBR","WBMG","PTMG","VISA","BTPL","CRWO","SNOC","RTAH","WTCR","SPSA","SUNB","FTHE","GBHE","GREG","SNEG","CAEG","GRHE","KIVU","BLVU","TUVU","OSPR","WTKI","HBKI","STKI","BLHE","ORHE","DTKI","COHA","BIHA","BAHA","ROHA","BWHA","STHA","SWHA","RTHA","STTR","GATR","COTR","LEMO","RMOT","AMKI","GKIN","RHBA","PBBA","NOET","COAR","YTTO","KBTO","BCWO","HOWO","SMBW","LIWO","GOWO","BAFF","LAFA","AMKE","BAFA","PEFA","RFPA","BAPA","BHOP","WCPA","SWPA","CFPA","RRAN","RUAN","CTAN","SLAN","ZEAN","SFTA","OLWO","PBRW","WBWO","SPWO","SHWO","PLXE","STXE","STFG","LIFG","BTFG","RUTR","RFSP","SLSP","YBEL","TOTY","OSTF","OBFL","SLCF","RBTY","RLTY","SCPT","COTF","BHTF","ERFL","YOFL","WTRS","RDTF","TCFL","TUFL","OSFL","DAPE","EAWP","TROP","YBFL","WTFL","LEFL","YEFL","BLPH","BRAT","RMOU","DCFL","GKIS","BOBF","SOFL","GCAF","GBFL","SBFL","PIRF","TRKI","LOCO","WRMA","WCOM","BCRT","MATI","NOSC","CIMB","RBPE","LESG","YTVI","PHVI","BCAV","BRJA","BAWS","NRWS","SRWS","HOWR","OCWR","BABW","BTWR","SIBW","CABW","BAYW","WBWW","GBWW","TRGN","BFSO","SWTH","WOTH","MOTH","PVTH","CCTH","GRCA","BAYS","GBCH","YCEU","YTEU","ELEU","OBAE","WVEU","TCEU","ATCH","COCL","BSTS","OBSP","CCBR","SFFI","RCOS","WNBR","CHOR","MORO","BCOR","BAOR","SHCO","BROC","GICO","MEBL","GTGR","LOWA","NOWA","GWWA","BWWA","BAWW","TEWA","GCYE","MOWA","OCYE","AMRE","TRPA","BLBW","YEWA","CSWA","YRWA","YTWA","BTNW","RCWA","GCRW","CRWA","BURW","WIWA","STRE","COLR","HETA","SUTA","WWTA","RTAT","CATA","BFAG","BTGG","INBU","WSTA","TCTA","WLTA","CCTA","BAGT","BGTA","PALT","SPTA","GHOT","SCHT","PCTA","BHTA","EMTA","STTA","STDA","GRHO","BAYT","SLFL","BGRA","TBSF","VASE","BANA","YFGR","BTSA","BHSA","GRAS"],
  },

  {
    name: 'La Mina near Rancho Naturalista',
    props: {r: 'L1882589', bmo: 12, emo: 2, byr: 2008},
    species: ["GRTI","GHEC","CRGU","RBPI","RUDP","SBPI","INDO","RUGD","RUQD","WTDO","WWDO","GBAN","SQCU","CCSW","WCSW","VASW","LSTS","WNJA","GREH","STHR","PCFA","GNBM","GRET","BCCO","GCBR","LBST","GAEM","VISA","BTPL","CRWO","SNOC","RTAH","RSWR","AMCO","PUGA","WTCR","SOLA","NOJA","SPSA","SUNB","ANHI","FTHE","GBHE","GREG","SNEG","LBHE","CAEG","GRHE","GRIB","KIVU","BLVU","TUVU","OSPR","WTKI","BLHE","BIHA","BAHA","ROHA","GRHA","BWHA","STHA","GATR","BTHT","COTR","LEMO","RMOT","RIKI","AMKI","GKIN","RTJA","RHBA","COAR","KBTO","BCWO","HOWO","SMBW","PBIW","LIWO","GOWO","YHCA","LAFA","MERL","RFPA","BAPA","BHOP","WCPA","RLPA","SWPA","CFPA","RRAN","BAAN","RUAN","PLAN","CTAN","SLAN","DMAN","ZEAN","BIAN","TTLE","PBRW","WBWO","COWO","SPWO","SHWO","PLXE","BTFG","SLSP","GREL","YBEL","TOTY","OSTF","OBFL","SLCF","SCPT","COTF","BHTF","YOFL","WTRS","ROFL","RDTF","TCFL","OSFL","TROP","YBFL","WTFL","LEFL","BLPH","LTTY","BRAT","RMOU","DCFL","GKIS","BOBF","SOFL","GCAF","PIRF","TRKI","WRMA","WCOM","BCRT","MATI","CIMB","WWBE","TCGR","LESG","YTVI","PHVI","BRJA","BAWS","NRWS","SRWS","BARS","SCBW","HOWR","BTWR","SIBW","CABW","BAYW","WBWW","TRGN","SWTH","WTTH","CCTH","YCEU","YTEU","OBAE","WVEU","TCEU","BSTS","OBSP","CCBR","RCOS","CHOR","MORO","SRCA","BCOR","BAOR","SHCO","BROC","GICO","MEBL","GTGR","LOWA","NOWA","GWWA","BAWW","PROW","TEWA","MOWA","OCYE","AMRE","TRPA","BBWA","BLBW","YEWA","CSWA","BTNW","RCWA","GCRW","BURW","WIWA","STRE","SUTA","WWTA","RTAT","BFAG","RBGR","WSTA","TCTA","WLTA","CCTA","BGTA","PALT","SPTA","GHOT","PCTA","BHTA","EMTA","STTA","STDA","GRHO","BGRA","TBSF","VASE","YBSE","BANA","YFGR","BTSA","BHSA","GRAS"],
  },

  {
    name: 'CATIE',
    props: {r: 'L1251289,L629014,L3994328', bmo: 12, emo: 2, byr: 2008},
    species: ["MUDU","BWTE","GHEC","BLAG","LEGR","RODO","PVPI","RBPI","BTPI","SBPI","RUGD","RUQD","WTDO","WWDO","GBAN","SQCU","MACU","LENI","COPA","WCSW","VASW","GRSW","LSTS","WNJA","GREH","STHR","GNBM","BCCO","RTHU","GAEM","VISA","CRWO","SNOC","SVHU","RTAH","RSWR","COGA","PUGA","WTCR","BNST","SOLA","NOJA","SPSA","WOST","ANHI","NECO","FTHE","BTTH","GBHE","GREG","SNEG","LBHE","TRHE","CAEG","GRHE","BCNH","YCNH","BBHE","GRIB","BLVU","TUVU","OSPR","WTKI","HBKI","GHKI","BLHE","SSHA","COHA","BIHA","ROHA","BWHA","STHA","ZTHA","FEPO","MOOW","GATR","LEMO","RIKI","AMKI","GKIN","NOET","COAR","YTTO","KBTO","BCWO","HOWO","SMBW","PBIW","LIWO","GOWO","CRCA","YHCA","LAFA","AMKE","BAFA","PEFA","OCPA","BHOP","WCPA","WFPA","OTPA","CFPA","FAAN","BAAN","BCAS","RUAN","CBAN","COWO","SHWO","SCRW","YETY","GREL","YBEL","OBFL","SLCF","COTF","BHTF","YOFL","YMFL","OSFL","EAWP","TROP","YBFL","WIFL","BLPH","LTTY","BRAT","DCFL","GCFL","GKIS","BOBF","SOFL","GCAF","WRFL","PIRF","TRKI","WRMA","WCOM","BCRT","MATI","CIMB","WWBE","LESG","YTVI","PHVI","BRJA","BAWS","NRWS","SRWS","GYBM","MANS","BANS","BARS","CLSW","HOWR","BABW","BTWR","CABW","BAYW","WBWW","LBGN","TRGN","SWTH","WOTH","CCTH","GRCA","TRMO","YCEU","YTEU","ELEU","OBAE","WVEU","TCEU","BSTS","CCSP","OBSP","RCOS","CAGS","EAME","RBBL","YBIC","CHOR","MORO","SRCA","BCOR","OROR","BAOR","RWBL","SHCO","BROC","GICO","MEBL","GTGR","OVEN","WEWA","NOWA","GWWA","BAWW","PROW","TEWA","GCYE","MOWA","OCYE","COYE","AMRE","TRPA","MAWA","BLBW","YEWA","CSWA","PAWA","YTWA","RCWA","GCRW","WIWA","COLR","SUTA","FCTA","WWTA","RBGR","BLGR","INBU","PABU","DICK","WSTA","WLTA","CCTA","BGTA","PALT","SPTA","GHOT","PCTA","BHTA","STTA","STDA","BLDA","RLHO","GRHO","BGRA","TBSF","VASE","YBSE","BANA","YFGR","BTSA","BHSA","GRAS","HOSP"],
  },

  {
    name: 'PN Volcán Irazú',
    props: {r: 'L441921', bmo: 12, emo: 2, byr: 2008},
    species: ["BLAG","BCWP","RODO","RBPI","BTPI","BFQD","MODO","DUNI","WCSW","VASW","GREH","LEVI","GCBR","TAHU","FTHU","PTMG","RTHU","VOHU","SCHU","VISA","RTAH","SNEG","CAEG","BLVU","TUVU","WTKI","SSHA","COHA","BWHA","STHA","RTHA","BSSO","CRPO","USWO","REQU","YBSA","ACWO","HOWO","HAWO","GOWO","MERL","PEFA","BAPA","SCRW","BUTU","RUTR","RFSP","YBEL","MOEL","RLTY","YBFL","YEFL","BCAF","BLPH","DCFL","GKIS","BOBF","SOFL","TRKI","YTVI","YWVI","PHVI","BAWS","NRWS","HOWR","OCWR","TIWR","CABW","GBWW","BFSO","BBNT","RCNT","MOTH","CCTH","SOOT","BAYS","LTSF","YBSI","SCCH","COCL","VOJU","RCOS","LFFI","WEGS","YTFI","WNBR","WRET","EAME","YBIC","BAOR","BROC","GTGR","BAWW","FTHW","TEWA","CSWA","BTNW","BCWA","WIWA","STRE","COLR","SUTA","FCTA","RBGR","BGTA","SCHT","SLFL","PBFI","BGRA","YFGR","HOSP"],
  },

  {
    name: 'PN Tapanti--Sector Tapanti',
    props: {r: 'L447854', bmo: 12, emo: 2, byr: 2008},
    species: ["BBWD","MUDU","BWTE","LESC","GHEC","CRGU","BLAG","BCWP","BBWQ","SPWQ","LEGR","RODO","RBPI","BTPI","RUDP","SBPI","RUGD","RUQD","WTDO","BFQD","CHQD","WWDO","GBAN","SQCU","COPA","DUNI","BLSW","WCHS","CCSW","WCSW","VASW","LSTS","WNJA","WTSI","GREH","STHR","GFRL","BRVI","LEVI","PCFA","GNBM","GRET","GCBR","TAHU","LBST","FTHU","WBMG","PTMG","WTMG","MTWO","VOHU","SCHU","GAEM","VHHU","VISA","CRWO","STHM","BLBH","CHEM","SNOC","SVHU","RTAH","WTCR","SOLA","KILL","SPSA","LEYE","SUNB","GBHE","GREG","SNEG","LBHE","CAEG","GRHE","GRIB","BLVU","TUVU","OSPR","WTKI","HBKI","STKI","BLHE","ORHE","DTKI","TIHA","COHA","BIHA","GBLH","BAHA","ROHA","GRHA","BWHA","STHA","RTHA","BSSO","CRPO","FEPO","MOOW","GATR","OBTR","COTR","LEMO","RIKI","AMKI","LAMO","RTJA","RHBA","PBBA","NOET","COAR","YETO","KBTO","ACWO","BCWO","HOWO","HAWO","SMBW","LIWO","GOWO","BAFF","CRCA","LAFA","AMKE","BAFA","RFPA","BAPA","OCPA","BHOP","WCPA","SWPA","CFPA","RRAN","BAAN","RUAN","PLAN","CTAN","SLAN","DWAN","DUAN","CBAN","ZEAN","BIAN","SPAN","SCAA","OBAN","SFTA","RBAN","TTLE","OLWO","LTWO","RUWO","PBRW","WBWO","BBNW","COWO","SPWO","BBSC","SHWO","SCRW","PLXE","STXE","BUTU","BFFG","STFG","LIFG","SBTR","BTFG","SPBA","RUTR","RFSP","SLSP","YBEL","LEEL","MOEL","TOTY","OSTF","OBFL","SLCF","RBTY","RLTY","SCPT","COTF","BHTF","ERFL","YOFL","YMFL","WTRS","GCRS","RDTF","SRFL","TUFL","OSFL","DAPE","WEWP","EAWP","TROP","YBFL","WTFL","YEFL","BLPH","LTTY","BRAT","RMOU","DCFL","GKIS","BOBF","SOFL","GCAF","WRFL","GBFL","SBFL","PIRF","TRKI","EAKI","SHAR","RUFP","WRMA","MATI","BABE","CIMB","BAWB","RBPE","TCGR","LESG","YTVI","PHVI","BCAV","AHJA","BRJA","BAWS","NRWS","SRWS","GYBM","MANS","BARS","NIWR","HOWR","OCWR","SEWR","BABW","CABW","BAYW","WBWW","GBWW","TRGN","AMDI","BFSO","OBNT","SBNT","RCNT","BHNT","SWTH","WOTH","MOTH","PVTH","WTTH","CCTH","TRMO","BAYS","LTSF","GBCH","YCEU","YTEU","ELEU","OBAE","WVEU","TCEU","LEGO","ATCH","SCCH","COCL","CCBR","SFFI","RCOS","WEGS","CAGS","YTFI","WNBR","WRET","EAME","RBBL","CHOR","MORO","BCOR","OROR","BAOR","BROC","GICO","MEBL","GTGR","OVEN","LOWA","NOWA","GWWA","BWWA","BAWW","PROW","FTHW","TEWA","GCYE","MOWA","OCYE","COYE","AMRE","TRPA","BBWA","BLBW","YEWA","CSWA","YRWA","YTWA","TOWA","BTNW","RCWA","BCWA","GCRW","CRWA","BURW","CAWA","WIWA","STRE","COLR","HETA","SUTA","WWTA","RTAT","CATA","BFAG","BTGG","RBGR","INBU","WSTA","WLTA","CCTA","BAGT","BGTA","PALT","SPTA","GHOT","SCHT","PCTA","RWTA","BHTA","EMTA","STTA","STDA","BLDA","GRHO","BAYT","SLFL","BGRA","TBSF","VASE","BANA","YFGR","BTSA","BHSA","GRAS","HOSP"],
  },

];