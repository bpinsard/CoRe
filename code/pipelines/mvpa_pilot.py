import os,sys,glob,datetime
import numpy as np

from nipype.interfaces import spm, fsl, afni, nitime, utility, lif, dcmstack as np_dcmstack, freesurfer, nipy, ants

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nibabel as nb

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
afni.base.AFNICommand.set_default_output_type('NIFTI_GZ')
np_dcmstack.DCMStackBase.set_default_output_type('NIFTI_GZ')

sys.path.insert(0,'/home/bpinsard/data/src/misc')
import generic_pipelines
from generic_pipelines.utils import wrap, fname_presuffix_basename, wildcard

from nipype import config
cfg = dict(execution={'stop_on_first_crash': False})
config.update_config(cfg)

data_dir = '/home/bpinsard/data/raw/UNF/CoRe'
mri_data_dir = os.path.join(data_dir,'MRI')
proc_dir = '/home/bpinsard/data/analysis/'

subjects = ['S00_BP_pilot','S01_ED_pilot']
subjects = subjects[1:]
#subjects = subjects[:1]


tr = 2.16
file_pattern = '_%(PatientName)s_%(SeriesDescription)s_%(SeriesDate)s_%(SeriesTime)s'
meta_tag_force=['PatientID','PatientName','SeriesDate']

def dicom_dirs():

    subjects_info = pe.Node(
        utility.IdentityInterface(fields=['subject']),
        name='subjects_info')
    subjects_info.iterables = [('subject', subjects)]

    anat_dirs = pe.Node(
        nio.DataGrabber(infields=['subject'],
                        outfields=['aa_scout','localizer',
                                   'fmri_resting_state','fmri_task1','fmri_task2',
                                   'fmri_fieldmap', 'fmri_pa'],
                        sort_filelist = True,
                        raise_on_empty = False,
                        base_directory = mri_data_dir, template=''),
        name='anat_dirs')
    anat_dirs.inputs.template = 'CoRe_%s_01/??-%s'
    ## all this will change with 32 channel head coil
    anat_dirs.inputs.template_args = dict(
        t1_mprage=[['subject','anat_mprage-MPRAGE_12ch']], 
        aa_scout=[['subject','AAScout']],
        localizer=[['subject','localizer_12Channel']],
        )

    func_dirs = pe.Node(
        nio.DataGrabber(infields=['subject', 'day'],
                        outfields=['aa_scout','localizer',
                                   'fmri_resting_state','fmri_task1','fmri_task2',
                                   'fmri_fieldmap', 'fmri_pa'],
                        sort_filelist = True,
                        raise_on_empty = False,
                        base_directory = mri_data_dir, template=''),
        name='func_dirs')
    func_dirs.inputs.template = 'CoRe_%s_%02d/??-%s'
    func_dirs.inputs.template_args = dict(
        aa_scout=[['subject','day','AAScout']],
        localizer=[['subject','day','localizer_12Channel']],
        fmri_resting_state=[['subject','day','BOLD_Resting_State']],
        fmri_all = [['subject','day','BOLD_*']],
        fmri_seqA=[['subject','day','BOLD_Task1']],
        fmri_seqB=[['subject','day','BOLD_Task2']],
        fmri_mvpa=[['subject','day','BOLD_MVPA?']],
        fmri_pa=[['subject','day','BOLD_PA']],
        fmri_fieldmap=[['subject','day','gre_field_map_BOLD/*']],)
    
    func_dirs.iterables = [('day',[1,2,3])] # [('day',[1,2,3])]
    
    n_all_func_dirs = pe.JoinNode(
        utility.IdentityInterface(fields=['fmri_all','fmri_fieldmap_all','fmri_pa_all']),
        joinsource = 'func_dirs',
        name='all_func_dirs')
    w = pe.Workflow(name='core')

    for n in [anat_dirs, func_dirs]:
        w.connect([(subjects_info,n,[('subject',)*2])])
    w.connect([
            (func_dirs, n_all_func_dirs,[('fmri_all',)*2,
                                         ('fmri_fieldmap','fmri_fieldmap_all',),
                                         ('fmri_pa','fmri_pa_all')]),
            ])        
    return w
    
def preproc_anat():
    
    w = dicom_dirs()

    n_t1_convert = pe.Node(
        np_dcmstack.DCMStackAnatomical(
            meta_force_add=meta_tag_force,
            out_file_format = 't1_mprage'+'_%(PatientName)s_%(SeriesDate)s_%(SeriesTime)s',
            voxel_order='LAS'),
        name='convert_t1_dicom')

    t1_pipeline = generic_pipelines.t1_new.t1_freesurfer_pipeline()
    t1_pipeline.inputs.freesurfer.args=''
    t1_pipeline.inputs.freesurfer.openmp = 8
    wm_surface = generic_pipelines.t1_new.extract_wm_surface()

    n_crop_t1 = pe.Node(
        nipy.Crop(out_file='%s_crop.nii.gz',
                  outputtype='NIFTI_GZ'),
        name='crop_t1')
    t1_pipeline.connect([
            (t1_pipeline.get_node('autobox_mask_fs'),n_crop_t1,
             [('%s_%s'%(c,p),)*2 for c in 'xyz' for p in ('min','max')]),
            (t1_pipeline.get_node('freesurfer'),n_crop_t1,
             [('norm','in_file')])])

    n_fs32k_surf = generic_pipelines.fmri_surface.surface_32k()

    n_reg_crop = pe.Node(
        freesurfer.Tkregister(reg_file='reg_crop.dat',
                              freeview='freeview.mat',
                              fsl_reg_out='fsl_reg_out.mat',
                              lta_out='lta.mat',
                              xfm_out='xfm.mat',
                              reg_header=True),
        'reg_crop')
    n_compute_pvmaps = pe.Node(
        freesurfer.ComputeVolumeFractions(args='--gm pve.gm.mgz'),
        name='compute_pvmaps')


    ants_for_sbctx = generic_pipelines.fmri_surface.ants_for_subcortical()
    ants_for_sbctx.inputs.inputspec.template = '/home/bpinsard/data/src/Pipelines/global/templates/MNI152_T1_1mm_brain.nii.gz'
    ants_for_sbctx.inputs.inputspec.coords = '/home/bpinsard/data/src/Pipelines/global/templates/91282_Greyordinates/Atlas_ROIs.csv'


    t1_pipeline.connect([
            (t1_pipeline.get_node('freesurfer'),n_reg_crop,[
                    ('norm','target'),('subject_id',)*2,('subjects_dir',)*2]),
            (t1_pipeline.get_node('autobox_mask_fs'),n_reg_crop,[
                    ('out_file','mov')]),
            (n_reg_crop, n_compute_pvmaps, [
                    ('reg_file',)*2]),
            (t1_pipeline.get_node('autobox_mask_fs'), n_compute_pvmaps, [
                    ('out_file','in_file')]),
            (t1_pipeline.get_node('freesurfer'),n_compute_pvmaps,[
                    ('subjects_dir',)*2]),            
            ])
    
    w.base_dir = proc_dir
    w.connect([
            (w.get_node('anat_dirs'),n_t1_convert,[('t1_mprage','dicom_files')]),
            (w.get_node('subjects_info'),t1_pipeline,[('subject','inputspec.subject_id')]),
            (n_t1_convert,t1_pipeline,[('nifti_file','inputspec.t1_file')]),
            (t1_pipeline, wm_surface, [(('freesurfer.aparc_aseg',utility.select,0),'inputspec.aseg')]),
            (t1_pipeline,n_fs32k_surf,[('freesurfer.subjects_dir','fs_source.base_directory'),]),
            (w.get_node('subjects_info'),n_fs32k_surf,[('subject','fs_source.subject')]),
            (t1_pipeline, ants_for_sbctx,[('crop_t1.out_file','inputspec.t1')]),
       ])

    return w


def preproc_fmri():
    
    w = dicom_dirs()
    templates = dict(
        subjects_dir=[['t1_preproc/_subject','subject','freesurfer']],
        norm = [['t1_preproc/_subject','subject','freesurfer/*[!e]/mri/norm.mgz']],
        white_matter_surface = [['extract_wm_surface/_subject','subject','surf_decimate/rlh.aparc+aseg_wm.nii_smoothed_mask.all']],
        cropped_mask = [['t1_preproc/_subject','subject','autobox_mask_fs/*.nii.gz']],
        cropped_t1 = [['t1_preproc/_subject','subject','crop_t1/*.nii.gz']],
        pve_maps = [
            ['t1_preproc/_subject','subject','compute_pvmaps/*.gm.mgz'],
            ['t1_preproc/_subject','subject','compute_pvmaps/*.wm.mgz'],
            ['t1_preproc/_subject','subject','compute_pvmaps/*.csf.mgz']],
        lowres_surf_lh = [
            ['surface_32k/_subject','subject',
             'white_resample_surf/mapflow/_white_resample_surf0/lh.white_converted.32k.gii'],
            ['surface_32k/_subject','subject',
             'pial_resample_surf/mapflow/_pial_resample_surf0/lh.pial_converted.32k.gii']],
        lowres_surf_rh = [
            ['surface_32k/_subject','subject',
             'white_resample_surf/mapflow/_white_resample_surf1/rh.white_converted.32k.gii'],
            ['surface_32k/_subject','subject',
             'pial_resample_surf/mapflow/_pial_resample_surf1/rh.pial_converted.32k.gii']],
        lowres_rois_coords = [['ants_for_subcortical/_subject','subject','coords_ants2fs/atlas_coords_fs.csv']],
        )
    n_anat_grabber = pe.Node(
        nio.DataGrabber(
            infields=['subject'],
            outfields=templates.keys(),
            sort_filelist=True,
            raise_on_empty=False,
            base_directory = proc_dir),
        name='anat_grabber')
    n_anat_grabber.inputs.template = 'core/%s_%s/%s'
    n_anat_grabber.inputs.template_args = templates


    n_fmri_convert = pe.MapNode(
        np_dcmstack.DCMStackfMRI(
            meta_force_add=meta_tag_force,
            out_file_format = 'fmri'+file_pattern),
        iterfield=['dicom_files'],
        name='fmri_convert')

    ## FIELDMAP PREPROC #####################################################################

    n_fieldmap_fmri_convert = pe.MapNode(
        np_dcmstack.DCMStackFieldmap(
            meta_force_add=meta_tag_force,
            out_file_format = 'fieldmap_bold'+file_pattern,
            voxel_order='LAS'),
        iterfield=['dicom_files'],
        name='convert_fieldmap_dicom')

    n_flirt_fmap2t1 = pe.MapNode(
        fsl.FLIRT(cost='mutualinfo',
                  dof=6,
                  out_matrix_file = 'fmap2t1.mat',
                  no_search=True,
                  uses_qform=True),
        iterfield=['in_file'],
        name='flirt_fmap2t1')

    n_flirt2xfm = pe.MapNode(
        freesurfer.preprocess.Tkregister(reg_file='reg.mat',
                                         xfm_out='fmap2t1.xfm'),
        iterfield=['mov','fsl_reg'],
        name='flirt2xfm')

    n_xfm2mat = pe.MapNode(
        utility.Function(
            input_names = ['xfm'],
            output_names = ['mat'],
            function=xfm2mat),
        iterfield=['xfm'],
        name='xfm2mat')

    def remove_none(l):
        return [e for e in l if e!=None]
    w.connect([
            (w.get_node('all_func_dirs'),n_fieldmap_fmri_convert,[(('fmri_fieldmap_all',remove_none), 'dicom_files')]),

            (n_fieldmap_fmri_convert, n_flirt_fmap2t1,[('magnitude_file','in_file')]),
            (n_anat_grabber, n_flirt_fmap2t1,[('cropped_t1','reference'),]),
            
            (n_flirt_fmap2t1, n_flirt2xfm,[('out_matrix_file','fsl_reg')]),
            (n_fieldmap_fmri_convert, n_flirt2xfm,[('magnitude_file','mov')]),
            (n_anat_grabber, n_flirt2xfm,[('cropped_t1','target'),]),

            (n_flirt2xfm, n_xfm2mat,[('xfm_out','xfm')]),
            ])

    ## Phase Inversion TOPUP APPA fieldmap ###############################################################

    n_extract_4vol_pa = pe.Node(
        nipy.Trim(begin_index=-4,
                  out_file='%s_4vol_pa'),
        name='extract_4vol_pa')

    n_merge_appa = pe.MapNode(
        utility.Merge(2),
        iterfield='',
        name='merge_appa')

#    w.connect([
#            (n_fmri_convert, n_extract_4vol_pa,[(('nifti_file',utility.select,-2),'in_file')]),
#            ])

    ###############################################    ###############################################
    def name_struct(f,n,*args):
        if isinstance(f,list):
            f = tuple(f)
        return [tuple([n,f]+list(args))]

    def repeater(l,n):
        return [l]*n

    n_group_hemi_surfs = pe.Node(
        utility.Merge(2),
        name='group_hemi_surfs')

    n_st_realign = pe.MapNode(
        nipy.SpaceTimeRealigner(
            tr=tr,
            slice_info=2,
            slice_times='ascending'),
        iterfield=['in_file'],
        name='st_realign')

    n_bbreg_epi = pe.MapNode(
        freesurfer.BBRegister(contrast_type='t2',reg_frame=0,registered_file=True,init='fsl'),
        iterfield=['source_file'],
        name='bbreg_epi')

    n_mask2epi= pe.Node(
        freesurfer.ApplyVolTransform(interp='nearest',inverse=True),
        name='mask2epi')

    n_epi2t1_bbreg2xfm = pe.MapNode(
        freesurfer.preprocess.Tkregister(xfm_out='fmap2t1.xfm',
                                         freeview='freeview.mat',
                                         fsl_reg_out='fsl_reg_out.mat',
                                         lta_out='lta.mat',),
        iterfield=['reg_file','mov'],
        name='epi2t1_bbreg2xfm')

    n_epi2t1_xfm2mat = pe.MapNode(
        utility.Function(
            input_names = ['xfm'],
            output_names = ['mat'],
            function=xfm2mat),
        iterfield=['xfm'],
        name='epi2t1_xfm2mat')

    def repeat_fieldmaps(fmri_scans, fieldmaps, fieldmap_regs):
        fmaps_out = []
        fmap_regs_out = []
        i = 0
        for sess in fmri_scans:
            if sess != None:
                fmaps_out += [fieldmaps[i]]*len(sess)
                fmap_regs_out += [fieldmap_regs[i]]*len(sess)
                i += 1
        return fmaps_out, fmap_regs_out
    n_repeat_fieldmaps = pe.Node(
        utility.Function(
            input_names = ['fmri_scans','fieldmaps','fieldmap_regs'],
            output_names = ['fieldmaps','fieldmap_regs'],
            function = repeat_fieldmaps),
        name='repeat_fieldmaps')

    n_convert_motion_par = generic_pipelines.fmri_surface.n_convert_motion_par

    n_noise_corr = pe.MapNode(
        nipy.preprocess.OnlineFilter(
            echo_time=.03,
            echo_spacing=.000265,
            phase_encoding_dir=1,
            resampled_first_frame='frame1.nii',
            out_file_format='ts.h5'),
        iterfield = ['dicom_files','motion','fieldmap','fieldmap_reg'],
        overwrite=False,
        name = 'noise_correction')

    n_smooth_bp = pe.MapNode(
        generic_pipelines.fmri_surface.GrayOrdinatesBandPassSmooth(
            data_field = 'FMRI/DATA',
            smoothing_factor=3,
            smoothing_steps=3,
            filter_range=(.008,.1),
            TR=2.16),
        iterfield = ['in_file'],
        name = 'smooth_bp')

    """
    n_surf_resample = pe.MapNode(
        nipy.preprocess.SurfaceResampling(
            echo_time=.03,
            echo_spacing=.000265,
            phase_encoding_dir=1,
            resampled_first_frame='frame1.nii',
            out_file_format='ts.h5'),
        iterfield = ['dicom_files','motion','fieldmap','fieldmap_reg'],
        overwrite=False,
        name = 'surf_resample')
    
    n_smooth_bp_nofilt = pe.MapNode(
        generic_pipelines.fmri_surface.GrayOrdinatesBandPassSmooth(
            data_field = 'FMRI/DATA',
            smoothing_factor=3,
            smoothing_steps=3,
            filter_range=(.008,.1),
            TR=2.16),
        iterfield = ['in_file'],
        name = 'smooth_bp_nofilt')
        """
    bold_seqs = ['fmri_resting_state','fmri_seqA','fmri_seqB','fmri_mvpa','fmri_pa']

    def flatten_remove_none(l):
        from nipype.interfaces.utility import flatten
        return flatten([e for e in l if e!=None])
    
    w.base_dir = proc_dir
    si = w.get_node('subjects_info')
    w.connect([
            (si, n_anat_grabber, [('subject',)*2]),

             (n_anat_grabber, n_group_hemi_surfs,[
                    (('lowres_surf_lh',name_struct,'CORTEX_LEFT'),'in1'),
                    (('lowres_surf_rh',name_struct,'CORTEX_RIGHT'),'in2')]),

            (w.get_node('all_func_dirs'),n_fmri_convert,[(('fmri_all',flatten_remove_none),'dicom_files')]),

            (n_fmri_convert,n_st_realign,[('nifti_file','in_file')]),
            (n_anat_grabber,n_bbreg_epi,[('subjects_dir',)*2]),
            (si,n_bbreg_epi,[('subject','subject_id')]),
            (n_st_realign,n_bbreg_epi,[('out_file','source_file')]),


#            (n_st_realign,n_bbreg2xfm,[(('out_file',generic_pipelines.utils.getitem_rec,0),'mov')]),
#            (n_bbreg_epi, n_bbreg2xfm,[('out_reg_file','reg_file')]),
#            (n_anat_grabber,n_bbreg2xfm,[('subjects_dir',)*2]),            
#            (n_bbreg_epi,n_convert_motion_par,[('out_reg_file','epi2t1')]),
            (n_st_realign,n_epi2t1_bbreg2xfm,[('out_file','mov')]),
            (n_bbreg_epi, n_epi2t1_bbreg2xfm,[('out_reg_file','reg_file')]),
            (n_anat_grabber,n_epi2t1_bbreg2xfm,[('norm','target')]),
            (n_epi2t1_bbreg2xfm, n_epi2t1_xfm2mat,[('xfm_out','xfm')]),
            

            (n_epi2t1_xfm2mat,n_convert_motion_par,[('mat','epi2t1')]),
            (n_st_realign, n_convert_motion_par,[('par_file','motion')]),
            (n_fmri_convert, n_convert_motion_par,[('nifti_file','matrix_file')]),

            (n_convert_motion_par, n_noise_corr,[('motion','motion')]),
            (w.get_node('all_func_dirs'),n_noise_corr,[(('fmri_all', flatten_remove_none),'dicom_files')]),

            (n_anat_grabber, n_noise_corr,[
                    ('norm','surfaces_volume_reference'),
                    ('cropped_mask','mask'),
                    (('lowres_rois_coords',name_struct,'SUBCORTICAL_CEREBELLUM',
                      '/home/bpinsard/data/projects/motion_correction/code/aparc.a2009s+aseg_subcortical_subset.txt'),
                     'resample_rois'),
                    ('pve_maps','partial_volume_maps')
                    ]),
            (w.get_node('all_func_dirs'),n_repeat_fieldmaps,[('fmri_all','fmri_scans')]),
            (n_fieldmap_fmri_convert, n_repeat_fieldmaps, [('fieldmap_file','fieldmaps')]),
            (n_xfm2mat, n_repeat_fieldmaps, [('mat','fieldmap_regs')]),
            (n_repeat_fieldmaps, n_noise_corr,[
                    ('fieldmaps','fieldmap'),
                    ('fieldmap_regs','fieldmap_reg')]),

            (n_group_hemi_surfs, n_noise_corr,[('out','resample_surfaces')]),
            (n_noise_corr, n_smooth_bp,[('out_file','in_file')]),
            
            ])
    if False:
        w.connect([
            (n_anat_grabber, n_surf_resample,[
                    ('norm','surfaces_volume_reference'),
                    ('cropped_mask','mask'),
                    (('lowres_rois_coords',name_struct,'SUBCORTICAL_CEREBELLUM',
                      '/home/bpinsard/data/projects/motion_correction/code/aparc.a2009s+aseg_subcortical_subset.txt'),
                     'resample_rois'),
                    ('pve_maps','partial_volume_maps')
                    ]),
            (n_repeat_fieldmaps, n_surf_resample,[
                    ('fieldmaps','fieldmap'),
                    ('fieldmap_regs','fieldmap_reg')]),


            (n_convert_motion_par, n_surf_resample,[('motion','motion')]),
            (w.get_node('all_func_dirs'), n_surf_resample,[(('fmri_all', flatten_remove_none),'dicom_files')]),
            (n_group_hemi_surfs, n_surf_resample,[('out','resample_surfaces')]),

            (n_surf_resample, n_smooth_bp_nofilt,[('out_file','in_file')]),            
            ])

    return w 


def xfm2mat(xfm):
    import os
    import numpy as np
    mat=np.loadtxt(xfm, skiprows=5,usecols=range(4))
    np.savetxt("out.mat",np.vstack([mat,[0,0,0,1]]))
    return os.path.abspath("out.mat")        

def grab_preproc(subject_id):
    import os, glob
    import numpy as np
    from mvpa_pilot import data_dir, proc_dir
    scans = np.loadtxt(os.path.join(data_dir,'Design/CoRe_%s.csv'%subject_id),delimiter=',',dtype=np.str)
    noise_corrected_ts = [os.path.join(
            proc_dir,'core',
            '_subject_%s/noise_correction/mapflow/_noise_correction%d/ts.h5'%(subject_id, int(i))) for i in scans[:,0]]
    smoothed_ts = [os.path.join(
            proc_dir,'core',
            '_subject_%s/smooth_bp/mapflow/_smooth_bp%d/ts_smooth_bp.h5'%(subject_id, int(i))) for i in scans[:,0]]
    seqs = [(None if s=='None' else s) for s in scans[:,2]]
    behs = [(None if s=='None' else s) for s in scans[:,3]]
    return noise_corrected_ts, smoothed_ts, scans[:,1].tolist(), seqs, behs

def mvpa_pipeline():
    w = dicom_dirs()

    n_preproc_grabber = pe.Node(
        utility.Function(
            input_names=['subject_id'],
            output_names=['noise_corrected_ts','smoothed_ts', 'session_names', 'sequence_names', 'behavior_files'],
            function=grab_preproc),
        name='preproc_grabber')

    n_dataset_noisecorr = pe.Node(
        CreateDataset(tr=tr),
        name='dataset_noisecorr')

    n_dataset_smoothed = pe.Node(
        CreateDataset(tr=tr),
        name='dataset_smoothed')

    w.base_dir = proc_dir
    si = w.get_node('subjects_info')
    w.connect([
            (si, n_preproc_grabber, [('subject','subject_id')]),
            (n_preproc_grabber, n_dataset_noisecorr, [('noise_corrected_ts','ts_files')]),
            (n_preproc_grabber, n_dataset_smoothed, [('smoothed_ts','ts_files')]),
            ])
    for n in [n_dataset_noisecorr, n_dataset_smoothed]:
        w.connect(si, 'subject', n, 'subject_id')
        for f in ['session_names','sequence_names','behavior_files']:
            w.connect(n_preproc_grabber, f, n, f)
    return w
    
from nipype.interfaces.base import (TraitedSpec, BaseInterface, traits,
                                    BaseInterfaceInputSpec, isdefined, File, Directory,
                                    InputMultiPath, OutputMultiPath)
import mvpa2.datasets

class CreateDatasetInputSpec(BaseInterfaceInputSpec):
    subject_id = traits.Str(mandatory=True)
    ts_files = traits.List(File(exists=True))
    data_path = traits.Str('FMRI/DATA',usedefault=True,)
    session_names = traits.List(traits.Str())
    sequence_names = traits.List(traits.Either(traits.Str(),None))
    behavior_files = traits.List(traits.Either(File(),None))

    tr = traits.Float(mandatory=True)

class CreateDatasetOutputSpec(TraitedSpec):
    dataset = File(exists=True, mandatory=True)
    glm_dataset = File(exists=True, mandatory=True)
    glm_stim_dataset = File(exists=True, mandatory=True)

class CreateDataset(BaseInterface):

    input_spec = CreateDatasetInputSpec
    output_spec = CreateDatasetOutputSpec

    def _run_interface(self, runtime):
        dss = []
        dss_glm = []
        dss_glm_stim = []
        scan_id = 0
        for ts_file, ses_name, seq_name, beh in zip(
            self.inputs.ts_files, self.inputs.session_names,
            self.inputs.sequence_names, self.inputs.behavior_files):
            seq_idx = None
            seq_info = None
            if not seq_name is None and not beh is None:
                seq_info = [('CoReTSeq', np.asarray([1,4,2,3,1])),
                            ('CoReIntSeq', np.asarray([1,3,2,4,1])),
                            ('mvpa_CoReOtherSeq', np.asarray([1,3,4,2,1])),
                            ('mvpa_CoreEasySeq', np.asarray([4,3,2,1,4]))]
                seq_idx = [[s[0] for s in seq_info].index(seq_name)] * 7
            ds = ds_from_ts(ts_file, beh, seq_info=seq_info, seq_idx=seq_idx, tr=self.inputs.tr)
            ds.sa['scan_name'] = [ses_name]*ds.nsamples
            ds.sa['scan_id'] = [scan_id]*ds.nsamples
            dss.append(ds)
            scan_id += 1
            if beh is not None:
                reg_groups = np.unique([n.split('_')[0] for n in ds.sa.regressors_exec.dtype.names])
                ds_glm = ds_tr2glm(ds, 'regressors_exec', reg_groups)
                dss_glm.append(ds_glm)
                ds_glm = ds_tr2glm(ds, 'regressors_stim', reg_groups)
                dss_glm_stim.append(ds_glm)

                del ds.sa['regressors_exec'], ds.sa['regressors_stim']
            
        # stack all
        ds = mvpa2.datasets.vstack(dss)
        ds.a.update(dss[0].a)
#        ds.a = dss[0].a
        ds.sa['chunks'] = np.cumsum(np.ediff1d(ds.chunks, to_begin=[0])!=0)
        ds_glm = mvpa2.datasets.vstack(dss_glm)
        ds_glm.a.update(dss[0].a)
        ds_glm.sa['chunks'] = np.cumsum(np.ediff1d(ds_glm.chunks, to_begin=[0])!=0)
        ds_glm_stim = mvpa2.datasets.vstack(dss_glm_stim)
        ds_glm_stim.a.update(dss[0].a)
        ds_glm_stim.sa['chunks'] = np.cumsum(np.ediff1d(ds_glm_stim.chunks, to_begin=[0])!=0)
        add_aparc_ba_fa(ds, self.inputs.subject_id, pproc_path='/home/bpinsard/data/analysis/core/')
        ds_glm.fa = ds.fa
        ds_glm_stim.fa = ds.fa
        
        outputs = self._list_outputs()
        ds.save(outputs['dataset'])
        ds_glm.save(outputs['glm_dataset'])
        ds_glm_stim.save(outputs['glm_stim_dataset'])

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['dataset'] = os.path.abspath('./ds_%s.h5'%self.inputs.subject_id)
        outputs['glm_dataset'] = os.path.abspath('./glm_ds_%s.h5'%self.inputs.subject_id)
        outputs['glm_stim_dataset'] = os.path.abspath('./glm_stim_ds_%s.h5'%self.inputs.subject_id)
        return outputs
        
import sys
sys.path.insert(0, '/home/bpinsard/data/projects/CoRe/code/')
from behavior import load_behavior

from nipy.modalities.fmri.experimental_paradigm import BlockParadigm, EventRelatedParadigm
from nipy.modalities.fmri.glm import GeneralLinearModel
from nipy.modalities.fmri.design_matrix import dmtx_light

def blocks_to_attributes(ds, blocks, hrf_rest_thresh=.2, tr=tr):
    
    instrs = np.asarray(
        [['instr_%03d_%s'%(bi,b[0]),b[0],b[2],b[3]-b[2],b[5]-b[2]] for bi,b in enumerate(blocks) if b[2]>0],
        dtype=np.object).reshape(-1,5)
    gos = np.asarray(
        [['go_%03d_%s'%(bi,b[0]),b[0],b[3],b[4]-b[3]] for bi,b in enumerate(blocks)],
        dtype=np.object)
    execs = np.asarray(
        [['exec_%03d_%s'%(bi,b[0]),b[0],b[5],b[6]-b[5]] for bi,b in enumerate(blocks)],
        dtype=np.object)
    
    ds.a['blocks_tr'] = np.round(np.asarray([b[2] for b in blocks])/tr).astype(np.int)
    ds.a['blocks_targets'] = [b[0] for b in blocks]
        
    par_exec = BlockParadigm(
        con_id = np.hstack([instrs[:,0], execs[:,0]]),
        onset = np.hstack([instrs[:,2], execs[:,2]]),
        duration = np.hstack([instrs[:,4], execs[:,3]]))

    par_stim = BlockParadigm(
        con_id = np.hstack([instrs[:,0], gos[:,0]]),
        onset = np.hstack([instrs[:,2], gos[:,2]]),
        duration = np.hstack([instrs[:,3], gos[:,3]]))
    
    frametimes = ds.sa.time-ds.sa.time[0]

    n_correct_sequences = np.asarray([sum([np.all(s['match']) for s in b[-1]]) for b in blocks]+[-1])
    n_failed_sequences = np.asarray([sum([np.all(~s['match']) for s in b[-1]]) for b in blocks]+[-1])
    rts = [np.diff(np.hstack([s['time'] for s in b[-1]])) for b in blocks]
    rts_stats = np.asarray([[rt.mean(), rt.std()] for rt in rts])
    
    mtx_exec, names_exec = dmtx_light(frametimes, par_exec, hrf_model='canonical', drift_model='blank')
    mtx_stim, names_stim = dmtx_light(frametimes, par_stim, hrf_model='canonical', drift_model='blank')

    ds.sa['regressors_exec'] = np.array(mtx_exec, dtype=[(n,np.float) for n in names_exec])
    ds.sa['regressors_stim'] = np.array(mtx_stim, dtype=[(n,np.float) for n in names_stim])

    targ_idx = np.argmax(mtx_exec[:,:-1],1)
    ds.sa['targets'] = np.asarray(['_'.join(names_exec[i].split('_')[2:]) for i in targ_idx])
    ds.sa.targets[mtx_exec[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh] = 'rest'
    ds.sa['subtargets'] = np.asarray([names_exec[i].split('_')[0] for i in targ_idx])
    ds.sa.subtargets[mtx_exec[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh] = 'rest'
    ds.sa['blocks_idx'] = np.asarray([int(names_exec[i].split('_')[1]) for i in targ_idx])
    ds.sa.blocks_idx[mtx_exec[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh] = -1
    ds.sa['sequence'] = np.asarray([''.join(blocks[i][1].astype(np.str)) for i in ds.sa.blocks_idx])

    ds.sa['n_correct_sequences'] = n_correct_sequences[ds.sa.blocks_idx]
    ds.sa['n_failed_sequences'] = n_failed_sequences[ds.sa.blocks_idx]

    targ_idx = np.argmax(mtx_stim[:,:-1],1)
    ds.sa['targets_stim'] = np.asarray(['_'.join(names_stim[i].split('_')[2:]) for i in targ_idx])
    ds.sa.targets_stim[mtx_stim[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh] = 'rest'
    ds.sa['subtargets_stim'] = np.asarray([names_stim[i].split('_')[0] for i in targ_idx])
    ds.sa.subtargets_stim[mtx_stim[(np.arange(len(frametimes)),targ_idx)]<hrf_rest_thresh] = 'rest'

    # add time from instruction
    ds.sa['delay_from_instruction'] = [np.nan]*ds.nsamples
    ds.sa['tr_from_instruction'] = [np.nan]*ds.nsamples
    ds.sa['delay_from_go'] = [np.nan]*ds.nsamples
    ds.sa['delay_from_first_key'] = [np.nan]*ds.nsamples

    last_vol = 0
    for instr,go,ex in zip(instrs, gos, execs):
        first_vol = int(np.round(instr[2]/tr+1e-4))
        prev_vol = last_vol
        last_vol = int(np.ceil((go[2]+go[3])/tr))
        ds.sa.delay_from_instruction[prev_vol:last_vol+1] = np.arange(last_vol-prev_vol+1)*tr - (first_vol-prev_vol)*tr
        ds.sa.tr_from_instruction[prev_vol:last_vol] = np.arange(last_vol-prev_vol) - (first_vol-prev_vol)
        
        first_vol = int(np.floor(go[2]/tr))
        ds.sa.delay_from_go[first_vol:last_vol] = np.arange(last_vol-first_vol)*tr + (go[2]-first_vol*tr)

        first_vol = int(np.floor(ex[2]/tr))
        last_vol = int(np.ceil((ex[2]+ex[3])/tr))
        ds.sa.delay_from_first_key[first_vol:last_vol] = np.arange(last_vol-first_vol)*tr + (ex[2]-first_vol*tr)

    
    
import h5py
import datetime 

seq_info = [('CoReTSeq',np.asarray([1,4,2,3,1])),('CoReIntSeq',np.asarray([1,3,2,4,1]))]
seq_idx = [0]*7

from mvpa2.datasets import Dataset
from mvpa2.mappers.detrend import poly_detrend

def ds_from_ts(ts_file, design_file=None,
               remapping=None, seq_info=None, seq_idx=None,
               default_target='rest', tr=tr, data_path='FMRI/DATA'):

    ts = h5py.File(ts_file,'r')    
    ds = Dataset(np.transpose(ts[data_path]))
    if np.count_nonzero(np.isnan(ds.samples)) > 0:
        print 'Warning : dataset contains NaN, replaced with 0 and created nans_mask'
        nans_mask = np.any(np.isnan(ds.samples), 0)
        ds.fa['nans'] = nans_mask
        ds.samples[:,nans_mask] = 0

    add_trend_chunk(ds)
    polyord = (np.bincount(ds.sa.trend_chunks)>(64./tr)).astype(np.int)
    poly_detrend(ds, chunks_attr='trend_chunks', polyord=polyord)
    
    ds.fa['coordinates'] = ts['COORDINATES'][:]
    ds.a['triangles'] = np.vstack([
            ts['STRUCTURES/CORTEX_LEFT/TRIANGLES'],
            ts['STRUCTURES/CORTEX_RIGHT/TRIANGLES'][:]+\
                (np.max(ts['STRUCTURES/CORTEX_LEFT/TRIANGLES'])+1)])
            
    ds.fa['node_indices'] = np.arange(ds.nfeatures,dtype=np.uint)
    if 'STRUCTURES/SUBCORTICAL_CEREBELLUM/INDICES' in ts:
        ds.fa['voxel_indices'] = np.empty((ds.nfeatures,3),dtype=np.int)
        ds.fa.voxel_indices.fill(np.nan)
        rois_offset = ts['STRUCTURES/SUBCORTICAL_CEREBELLUM/ROIS'][0,'IndexOffset']
        ds.fa.voxel_indices[rois_offset:] = ts['STRUCTURES/SUBCORTICAL_CEREBELLUM/INDICES']

    if 'scan_time' in ts[data_path].attrs and 'scan_date' in ts[data_path].attrs:
        date = ts[data_path].attrs['scan_date']
        time = ts[data_path].attrs['scan_time']
        dt = datetime.datetime.strptime(date+':'+time,'%Y%m%d:%H%M%S.%f')
        tstp = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
        ds.sa['time'] = tstp+np.arange(ds.nsamples)*tr
    else:
        ds.sa['time'] = np.arange(ds.nsamples)*tr

    target_chunk_len = 6
    if not design_file is None:
        blocks = load_behavior(
            design_file,
            remapping=remapping,
            seq_info=seq_info,
            seq_idx=seq_idx)
        blocks_to_attributes(ds, blocks, tr=tr)
        ds.sa['chunks'] = np.hstack([[0],np.cumsum(ds.sa.targets[:-1]!=ds.sa.targets[1:])])
        chunks_count = np.bincount(ds.chunks)
        for chk in np.where(chunks_count>2*target_chunk_len)[0]:
            ds.chunks[ds.chunks==chk] = chk+(np.arange(target_chunk_len)*1000).repeat(
                int(np.ceil(chunks_count[chk]/float(target_chunk_len))))[:chunks_count[chk]]
        ds.sa['chunks'] = np.cumsum(np.ediff1d(ds.chunks,to_begin=[0])!=0)
    else:
        ds.sa['chunks'] = np.arange(target_chunk_len).repeat(int(ds.nsamples/float(target_chunk_len))+1)[:ds.nsamples]
        ds.sa['targets'] = [default_target]*ds.nsamples
        ds.sa['subtargets'] = ds.sa.targets
        ds.sa['targets_stim'] = ds.sa.targets
        ds.sa['subtargets_stim'] = ds.sa.targets
        ds.sa['sequence'] = ['']*ds.nsamples


        for attr in ['n_correct_sequences',
                     'n_failed_sequences',
                     'delay_from_instruction',
                     'delay_from_first_key',
                     'delay_from_go',
                     'tr_from_instruction',
                     'blocks_idx',]:
            ds.sa[attr] = [np.nan]*ds.nsamples
    return ds

def add_aparc_ba_fa(ds, subject, pproc_path):
    roi_aparc = np.loadtxt('/home/bpinsard/data/src/Pipelines/global/templates/91282_Greyordinates/Atlas_ROIs.csv',
                           skiprows=1,delimiter=',')[:,-1].astype(np.int)
    
    aparcs_surf = np.hstack([nb.gifti.read(pproc_path+'surface_32k/_subject_%s/label_resample/mapflow/_label_resample%d/%sh.aparc.a2009s.annot_converted.32k.gii'%(subject,i,h)).darrays[0].data.astype(np.int)+11100+i*1000 for i,h in enumerate('lr')])
    ds.fa['aparc'] = np.hstack([aparcs_surf, roi_aparc])
        
    ba_32k = np.hstack([nb.gifti.read(pproc_path+'surface_32k/_subject_%s/BA_resample/mapflow/_BA_resample%d/%sh.BA_exvivo.annot_converted.32k.gii'%(subject,i,h)).darrays[0].data.astype(np.int) for i,h in enumerate('lr')] + [np.zeros(len(roi_aparc))])
    ba_thresh_32k = np.hstack([nb.gifti.read(pproc_path+'surface_32k/_subject_%s/BA_thresh_resample/mapflow/_BA_thresh_resample%d/%sh.BA_exvivo.thresh.annot_converted.32k.gii'%(subject,i,h)).darrays[0].data.astype(np.int) for i,h in enumerate('lr')] + [np.zeros(len(roi_aparc))])
    ds.fa['ba'] = ba_32k
    ds.fa['ba_thres'] = ba_thresh_32k

def add_trend_chunk(ds):
    ds.sa['trend_chunks'] = np.zeros(ds.nsamples)
    min_trend_chunk_len = 32./tr
    newchunk = np.zeros(ds.nsamples,dtype=np.bool)
    diffmean = np.mean(np.abs(np.diff(ds.samples,1,0)),1)
    diffmean = np.hstack([0,diffmean])
    cutoff = diffmean.mean()+2*diffmean.std()
    while True:
        c = np.argmax(diffmean)
        if diffmean[c] < cutoff:
            break
        tc = ds.sa.trend_chunks[c]
        cm = ds.sa.trend_chunks==tc
        if c > 0 and np.count_nonzero(cm[:c])>min_trend_chunk_len and np.count_nonzero(cm[c:])>min_trend_chunk_len:
            around = diffmean < cutoff
            newchunk[:] = np.logical_and(cm,np.arange(len(cm))>=c)
            ds.sa.trend_chunks[newchunk] = ds.sa.trend_chunks.max()+1
        diffmean[c] = 0
                
    ds.sa.trend_chunks = np.cumsum(np.ediff1d(ds.sa.trend_chunks,to_begin=[0])>0)

def ds_tr2glm(ds, regressors_attr, group_regressors):
    
    betas = []
    max_ind = []
    targets = []
    for reg_i, reg_name in enumerate(ds.sa[regressors_attr].value.dtype.names[:-1]):
        print 'fitting %s'%reg_name
        max_ind.append(np.argmax(ds.sa[regressors_attr].value[:,reg_i].astype(np.float)))

        summed_regs = np.asarray([ds.sa[regressors_attr].value.astype(np.float)[:,np.asarray([(n.split('_')[0]==rt and n!=reg_name) for n in  ds.sa[regressors_attr].value.dtype.names])].sum(1) for rt in group_regressors]).T
        mtx = np.hstack([ds.sa[regressors_attr].value[:,reg_i,np.newaxis].astype(np.float), summed_regs])
        glm = GeneralLinearModel(mtx)
        glm.fit(ds.samples)
        betas.append(np.squeeze(glm.get_beta(0)))
        del glm, mtx, summed_regs
        
    ds_glm = Dataset(np.asarray(betas), fa=ds.fa, a=ds.a)
    for attr in ['targets','subtargets','time','scan_id','scan_name']:
        ds_glm.sa[attr] = ds.sa[attr].value[max_ind]

    ds_glm.sa['chunks'] = np.arange(ds_glm.nsamples)
    return ds_glm

