import os,sys,glob,datetime
import numpy as np
import operator

from nipype.interfaces import spm, fsl, afni, nitime, utility, dcmstack as np_dcmstack, freesurfer, nipy, ants

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nibabel as nb
import dicom
import datetime

from ..mvpa import dataset as mvpa_dataset
import mvpa2.datasets

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
afni.base.AFNICommand.set_default_output_type('NIFTI_GZ')
np_dcmstack.DCMStackBase.set_default_output_type('NIFTI_GZ')

sys.path.insert(0,'/home/bpinsard/data/src/misc')
import generic_pipelines
import generic_pipelines.t1_new, generic_pipelines.fmri, generic_pipelines.fmri_surface
from generic_pipelines.utils import wrap, fname_presuffix_basename, wildcard
from generic_pipelines import moco_eval

from nipype import config
cfg = dict(execution={'stop_on_first_crash': False})
config.update_config(cfg)

data_dir = '/home/bpinsard/data/raw/UNF/CoRe'
mri_data_dir = os.path.join(data_dir,'MRI')
proc_dir = '/home/bpinsard/data/analysis/'
project_dir = '/home/bpinsard/data/projects/CoRe'


SEQ_INFO = [('CoReTSeq', np.asarray([1,4,2,3,1])),
            ('CoReIntSeq', np.asarray([1,3,2,4,1])),
            ('mvpa_CoReOtherSeq1', np.asarray([1,2,4,3,1])),
            ('mvpa_CoReOtherSeq2', np.asarray([4,1,3,2,4]))]


subject_ids = [1,11,23,22,63,50,67,79,54,107,128,162,102,82,155,100,94,87,192,200,184,194,195,220,223,235,256,268,267,237,283,296,319]
group_Int = [1,23,63,79,82,87,100,107,128,192,195,220,223,235,268,267,237,296]
#subject_ids=[67]
#subject_ids = subject_ids[:1]
#subject_ids = subject_ids[1:]
#subject_ids = [1,11,23,22,63,50,296]

tr = 2.16
echo_time = .03
echo_spacing = .00053
phase_encoding_dir = -1
middle_surface_position = .5
file_pattern = '_%(PatientName)s_%(SeriesDescription)s_%(SeriesDate)s_%(SeriesTime)s'
meta_tag_force=['PatientID','PatientName','SeriesDate','SeriesTime']

high_mem_queue_args = {'qsub_args': '-q high_mem', 'overwrite': True}
pe_queue_args = {'qsub_args': '-q parallel -pe smp 6', 'overwrite': True}

running_args = dict(
    plugin='SGE', 
    plugin_args={'template':os.path.join(os.path.dirname(__file__),'tpl.sh'),
                 'qsub_args': '-q default'}
)

def dicom_dirs():

    subjects_info = pe.Node(
        utility.IdentityInterface(fields=['subject_id']),
        run_without_submitting=True,
        name='subjects_info')
    subjects_info.iterables = [('subject_id', subject_ids)]

    anat_dirs = pe.Node(
        nio.DataGrabber(infields=['subject_id'],
                        outfields=['aa_scout','localizer','t1_mprage','t1_mprage_all_echos','t1_mprage_12ch_2mm_iso'],
                        sort_filelist = True,
                        raise_on_empty = False,
                        base_directory = mri_data_dir, template=''),
        run_without_submitting=True,
        name='anat_dirs')
    anat_dirs.inputs.template = 'CoRe*_%03d*/??-%s'
    ## all this will change with 32 channel head coil
    anat_dirs.inputs.template_args = dict(
        t1_mprage=[['subject_id','MEMPRAGE_4e_p2_1mm_iso RMS/*']],
        t1_mprage_all_echos=[['subject_id','MEMPRAGE_4e_p2_1mm_iso/*']],
        aa_scout=[['subject_id','AAScout']],
        localizer=[['subject_id','localizer_32Channel']],
        t1_mprage_12ch_2mm_iso=[['subject_id','MPRAGE_12ch_ipatx2_2x2x2']],
        )

    func_dirs = pe.Node(
        nio.DataGrabber(infields=['subject_id', 'day'],
                        outfields=['aa_scout','localizer',
                                   'fmri_resting_state','fmri_task1','fmri_task2',
                                   'fmri_fieldmap', 'fmri_pa'],
                        sort_filelist = True,
                        raise_on_empty = False,
                        base_directory = mri_data_dir, template=''),
        run_without_submitting=True,
        name='func_dirs')
    func_dirs.inputs.template = 'CoRe_%03d_D%d/??-*%s'
    func_dirs.inputs.template_args = dict(
        aa_scout=[['subject_id','day','AAScout']],
        localizer=[['subject_id','day','localizer_12Channel']],
        fmri_resting_state=[['subject_id','day','BOLD_Resting_State']],
        fmri_all = [['subject_id','day','BOLD_*']],
        fmri_ap = [['subject_id','day','BOLD_[!(PA)]*']],
        fmri_pa=[['subject_id','day','BOLD_PA']],
        fmri_fieldmap=[['subject_id','day','gre_field_map_BOLD/*']],)

    func_dirs.iterables = [('day',[1,2,3])]

    n_all_func_dirs = pe.JoinNode(
        utility.IdentityInterface(fields=['fmri_all','fmri_fieldmap_all','fmri_ap_all','fmri_pa_all','aa_scout_all']),
        joinsource = 'func_dirs',
        run_without_submitting=True,
        name='all_func_dirs')
    w = pe.Workflow(name='core_sleep')
    
    for n in [anat_dirs, func_dirs]:
        w.connect([(subjects_info,n,[('subject_id',)*2])])
    w.connect([
        (func_dirs, n_all_func_dirs,[('fmri_all',)*2,
                                     ('fmri_fieldmap','fmri_fieldmap_all',),
                                     ('fmri_pa','fmri_pa_all'),
                                     ('fmri_ap','fmri_ap_all'),
                                     ('aa_scout','aa_scout_all')]),
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
    
    
    n_t1_12ch_2mm_iso_convert = pe.Node(
        np_dcmstack.DCMStackAnatomical(
            meta_force_add=meta_tag_force,
            out_file_format = 't1_mprage_12ch_2mm_iso'+'_%(PatientName)s_%(SeriesDate)s_%(SeriesTime)s',
            voxel_order='LAS'),
        name='convert_t1_12ch_2mm_iso_dicom')
    

    n_n4_12ch_2mm = pe.Node(
        ants.segmentation.N4BiasFieldCorrection(
            dimension=3),
        name='n4_12ch_2mm')
    

    t1_pipeline = generic_pipelines.t1_new.t1_freesurfer_pipeline()
    t1_pipeline.inputs.freesurfer.args='-use-gpu'
    t1_pipeline.inputs.freesurfer.openmp = 8
    wm_surface = generic_pipelines.t1_new.extract_wm_surface()
    t1_pipeline.get_node('compute_pvmaps').plugin_args = high_mem_queue_args

    n_fs32k_surf = generic_pipelines.fmri_surface.surface_32k()
    
#    ants_for_sbctx = generic_pipelines.fmri_surface.ants_for_subcortical()
#    ants_for_sbctx.inputs.inputspec.template ='/home/bpinsard/data/src/Pipelines/global/templates/MNI152_T1_1mm_brain.nii.gz'
#    ants_for_sbctx.inputs.inputspec.coords = os.path.join(generic_pipelines.__path__[0],'../data','Atlas_ROIs.csv')
    warp_subctx = generic_pipelines.fmri_surface.warp_subcortical()
    warp_subctx.inputs.inputspec.template = os.path.join(generic_pipelines.__path__[0],'../data','MNI152_T1_1mm_brain_fslmask.nii.gz')
    warp_subctx.inputs.inputspec.coords = os.path.join(generic_pipelines.__path__[0],'../data','Atlas_ROIs.csv')

    w.base_dir = proc_dir
    w.connect([
        (w.get_node('anat_dirs'),n_t1_convert,[('t1_mprage','dicom_files')]),
#        (w.get_node('anat_dirs'),n_t1_12ch_2mm_iso_convert,[('t1_mprage_12ch_2mm_iso','dicom_files')]),
#        (n_t1_12ch_2mm_iso_convert, n_n4_12ch_2mm,[('nifti_file','input_image')]),
        (w.get_node('subjects_info'),t1_pipeline,[(('subject_id',wrap(str),[]),'inputspec.subject_id')]),
        (n_t1_convert,t1_pipeline,[('nifti_file','inputspec.t1_files')]),
        #(t1_pipeline, wm_surface, [(('freesurfer.aparc_aseg',utility.select,0),'inputspec.aseg')]),
        (t1_pipeline,n_fs32k_surf,[('freesurfer.subjects_dir','fs_source.base_directory'),]),
        (w.get_node('subjects_info'),n_fs32k_surf,[('subject_id','fs_source.subject')]),
        #(t1_pipeline, ants_for_sbctx,[('crop_brain.out_file','inputspec.t1')]),
        (t1_pipeline, warp_subctx,[('crop_brain.out_file','inputspec.t1')]),
    ])

    return w

def flatten_remove_none(l):
    from nipype.interfaces.utility import flatten
    return flatten([e for e in l if e!=None])

def group_by_3(lst):
    return reduce(lambda l,x: (l+[x[i:i+3] for i in range(0,len(x),3)] if x is not None else l),lst,[])

def name_struct(f,n,*args):
    if isinstance(f,list):
        f = tuple(f)
    return [tuple([n,f]+list(args))]
        
def repeater(l,n):
    return [l]*n
    

def preproc_fmri():
    
    w = dicom_dirs()
    si = w.get_node('subjects_info')

    templates = dict(
        subjects_dir=[['t1_preproc/_subject_id','subject_id','freesurfer']],
        norm = [['t1_preproc/_subject_id','subject_id','freesurfer/*[!e]/mri/norm.mgz']],
        white_matter_surface = [['extract_wm_surface/_subject_id','subject_id','surf_decimate/rlh.aparc+aseg_wm.nii_smoothed_mask.all']],
        cropped_mask = [['t1_preproc/_subject_id','subject_id','autobox_mask_fs/*.nii.gz']],
        cropped_t1 = [['t1_preproc/_subject_id','subject_id','crop_t1/*.nii.gz']],
        pve_maps = [
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.cortex.nii.gz'],
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.subcort_gm.nii.gz'],
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.wm.nii.gz'],
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.csf.nii.gz'],
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/pve.nii.gz']],
        lowres_surf_lh = [
            ['surface_32k/_subject_id','subject_id',
             'white_resample_surf/mapflow/_white_resample_surf0/lh.white_converted.32k.gii'],
            ['surface_32k/_subject_id','subject_id',
             'pial_resample_surf/mapflow/_pial_resample_surf0/lh.pial_converted.32k.gii']],
        lowres_surf_rh = [
            ['surface_32k/_subject_id','subject_id',
             'white_resample_surf/mapflow/_white_resample_surf1/rh.white_converted.32k.gii'],
            ['surface_32k/_subject_id','subject_id',
             'pial_resample_surf/mapflow/_pial_resample_surf1/rh.pial_converted.32k.gii']],
        lowres_rois_coords = [['warp_subcortical/_subject_id','subject_id','warp_coords/Atlas_ROIs_warped.csv']],
        eeg_coords = [['_subject_id','subject_id','warp_eeg_coords/grid_eeg_loc_mni_warped.csv']],
        warp2mni = [['warp_subcortical/_subject_id','subject_id','warp_to_mni/norm_crop_warp.npz']],
        #lowres_rois_coords = [['ants_for_subcortical/_subject_id','subject_id','coords_itk2nii/atlas_coords_nii.csv']],
        )
    n_anat_grabber = pe.Node(
        nio.DataGrabber(
            infields=['subject_id'],
            outfields=templates.keys(),
            sort_filelist=True,
            raise_on_empty=False,
            base_directory = proc_dir),
        run_without_submitting=True,
        name='anat_grabber')
    n_anat_grabber.inputs.template = 'core_sleep/%s_%s/%s'
    n_anat_grabber.inputs.template_args = templates


    n_fmri_convert = pe.MapNode(
        np_dcmstack.DCMStackfMRI(
            meta_force_add=meta_tag_force,
            out_file_format = 'fmri'+file_pattern),
        iterfield=['dicom_files'],
        name='fmri_convert')
    n_fmri_convert.plugin_args = high_mem_queue_args

    ## FIELDMAP PREPROC #####################################################################

    use_fieldmap=False
    if use_fieldmap:
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
            run_without_submitting=True,
            name='flirt2xfm')
        
        n_xfm2mat = pe.MapNode(
            utility.Function(
                input_names = ['xfm'],
                output_names = ['mat'],
                function=xfm2mat),
            iterfield=['xfm'],
            run_without_submitting=True,
            name='xfm2mat')

        w.connect([
            (w.get_node('all_func_dirs'),n_fieldmap_fmri_convert,[(('fmri_fieldmap_all',group_by_3), 'dicom_files')]),
            
            (n_fieldmap_fmri_convert, n_flirt_fmap2t1,[('magnitude_file','in_file')]),
            (n_anat_grabber, n_flirt_fmap2t1,[('cropped_t1','reference'),]),
            
            (n_flirt_fmap2t1, n_flirt2xfm,[('out_matrix_file','fsl_reg')]),
            (n_fieldmap_fmri_convert, n_flirt2xfm,[('magnitude_file','mov')]),
            (n_anat_grabber, n_flirt2xfm,[('cropped_t1','target'),]),

            (n_flirt2xfm, n_xfm2mat,[('xfm_out','xfm')]),
        ])

    ## Phase Inversion TOPUP APPA fieldmap ###############################################################

    n_merge_appa = generic_pipelines.fmri.merge_appa()
    
    n_topup = pe.MapNode(
        fsl.TOPUP(),
        iterfield=['in_file'],
        name='topup')
    n_topup.inputs.readout_times = [echo_spacing*(64-1)]*8
    n_topup.inputs.encoding_direction = ['x-']*4+['x']*4

    n_repeat_topup = pe.Node(
        utility.Function(
            input_names = ['fmri_scans','fieldmaps','arg_names','enc_file','appa','movpar','reg_file','pve'],
            output_names = ['fieldmaps','enc_file','appa','movpar','reg_file','pve'],
            function = repeat_fieldmaps),
        run_without_submitting=True,
        name='repeat_topup')
    n_repeat_topup.inputs.arg_names = n_repeat_topup.interface._input_names[3:]

    n_mcflirt = pe.MapNode(
        fsl.MCFLIRT(
            ref_vol=0,
            save_plots=True,
        ),
        iterfield = ['in_file','ref_file'],
        name='mcflirt')

    n_applytopup = pe.MapNode(
        fsl.ApplyTOPUP(
            in_index=[1],
            method='jac'),
        iterfield=['in_files','in_topup_fieldcoef','in_topup_movpar','encoding_file'],
        name='applytopup')
    n_applytopup.plugin_args = high_mem_queue_args


    n_apply_registration = pe.MapNode(
        utility.Function(
            input_names = ['in_file','matrix'],
            output_names = ['out_file'],
            function = generic_pipelines.fmri_surface.apply_affine),
        iterfield = ['in_file','matrix'],
        name='apply_registration'
    )

    n_epi_pvf = pe.MapNode(
        freesurfer.utils.ComputeVolumeFractions(
            gm_file='pve.nii.gz',
            niigz=True),
        iterfield = ['in_file','reg_file'],
        name='epi_pvf'
    )


    n_merge_gii = pe.Node(
        utility.Function(
            input_names = ['lh_tss','rh_tss','sc_tss'],
            output_names = ['grouped_tss'],
            function_str = 'def merge_gii(lh_tss,rh_tss,sc_tss): return [(l,r,s) for l,r,s in zip(lh_tss,rh_tss,sc_tss)]',
        ),
        name='merge_gii'
    )

    n_bbreg_epi_topup = pe.MapNode(
        freesurfer.BBRegister(
            contrast_type='t2',
            reg_frame=0,
            registered_file=True,
            init='fsl',
        ),
        iterfield=['source_file'],
        name='bbreg_epi_topup')
    

    n_good_voxels = pe.MapNode(
        utility.Function(
            input_names = ['in_file', 'mask_file', 'mask_threshold', 'factor'],
            output_names = ['good_voxels'],
            function=generic_pipelines.fmri_surface.fmri_goodvox),
        iterfield=['in_file','mask_file'],
        name='good_voxels')

    n_topup2t1_xfm = pe.MapNode(
        freesurfer.preprocess.Tkregister(
            xfm_out='topup2t1.xfm',
            freeview='freeview.mat',
            fsl_reg_out='fsl_reg_out.mat',
            lta_out='lta.mat',
            no_edit=True),
        iterfield=['reg_file','mov'],
        run_without_submitting=True,
        name='topup2t1_xfm')

    n_topup2t1_mat = pe.MapNode(
        utility.Function(
            input_names = ['xfm'],
            output_names = ['mat'],
            function=xfm2mat),
        iterfield=['xfm'],
        run_without_submitting=True,
        name='topup2t1_mat')

    n_dataset_wb = pe.Node(
        CreateDatasetWB(tr=tr,
                        behavioral_data_path=os.path.join(data_dir,'Behavior'),
                        #design=os.path.join(project_dir,'data/mvpa_only.csv'),
                        design=os.path.join(project_dir,'data/design.csv'),
                        #median_divide=True,
                        #wavelet_despike=True
                        hptf=True,
                        hptf_thresh=4
                    ),
        name='dataset_wb_hptf')
    n_dataset_wb.plugin_args = high_mem_queue_args


    topup = True
    if topup:

        w.connect([
            (n_fmri_convert, n_merge_appa,[('nifti_file','in_files')]),
            (n_merge_appa, n_topup,[('out_files','in_file')]),
            
            (n_topup, n_bbreg_epi_topup, [('out_corrected','source_file')]),
            (n_anat_grabber, n_bbreg_epi_topup, [('subjects_dir',)*2]),
            (si, n_bbreg_epi_topup, [(('subject_id',wrap(str),[]),'subject_id')]),            


            (n_anat_grabber, n_topup2t1_xfm, [('subjects_dir',)*2]),
            (n_topup,n_topup2t1_xfm,[('out_corrected','mov')]),
            (n_bbreg_epi_topup, n_topup2t1_xfm,[('out_reg_file','reg_file')]),
            #(n_anat_grabber,n_topup2t1_xfm,[('norm','target')]),
            (n_topup2t1_xfm, n_topup2t1_mat,[('xfm_out','xfm')]),
        
            #(n_bbreg_epi_topup, n_epi_pvf, [('out_reg_file','reg_file')]),
            #(n_topup, n_epi_pvf, [('out_corrected','in_file')]),
            #(n_anat_grabber,n_epi_pvf,[ ('subjects_dir',)*2]),

            #(n_epi_pvf,n_repeat_topup,[('partial_volume_maps','pve')]),
                
            (n_topup2t1_mat, n_repeat_topup,[('mat','reg_file')]),
            (w.get_node('all_func_dirs'),n_repeat_topup,[('fmri_all','fmri_scans')]),
            (n_topup, n_repeat_topup,[
                ('out_fieldcoef', 'fieldmaps'),
                ('out_movpar', 'movpar'),
                ('out_enc_file', 'enc_file')]),
            (n_merge_appa, n_repeat_topup,[('out_files','appa')]),

        ])

    ###############################################    ###############################################

#    n_convert_motion_par_scale = generic_pipelines.fmri_surface.n_convert_motion_par.clone('convert_motion_par_scale')    

    use_topup_fieldmap = True
    if use_topup_fieldmap:
        w.connect([
            (n_fmri_convert, n_mcflirt, [('nifti_file','in_file')]),
            (n_repeat_topup, n_mcflirt, [('appa','ref_file')]), #coregister to appa file closest

            (n_mcflirt, n_applytopup,[('out_file','in_files')]),
            (n_repeat_topup, n_applytopup,[
                ('fieldmaps','in_topup_fieldcoef'),
                ('movpar','in_topup_movpar'),
                ('enc_file','encoding_file'),]),
            (n_applytopup, n_apply_registration,[('out_corrected','in_file')]),
            (n_repeat_topup, n_apply_registration,[('reg_file','matrix')]),

        ])

    workbench_interpolate = False
    if workbench_interpolate:
        wb_pipe = generic_pipelines.fmri_surface.workbench_pipeline()
        w.connect([
            (n_anat_grabber,wb_pipe,[
                ('lowres_rois_coords','inputspec.lowres_rois_coords'),
                ('lowres_surf_lh','inputspec.lowres_surf_lh'),
                ('lowres_surf_rh','inputspec.lowres_surf_rh'),
            ]),
            (n_apply_registration, wb_pipe,[('out_file','inputspec.in_files')]),
        #(n_repeat_topup, n_good_voxels,[(('pve',generic_pipelines.utils.getitem_rec,slice(0,None),0),'mask_file')]),
        #(n_apply_registration, n_good_voxels,[('out_file','in_file')]),
            
            (si, n_dataset_wb,[('subject_id',)*2]),
            (wb_pipe, n_dataset_wb,[('merge_gii.grouped_tss','ts_files')]),
            (w.get_node('all_func_dirs'), n_dataset_wb,[(('fmri_all',flatten_remove_none),'dicom_dirs')]),
            
            (n_anat_grabber, n_dataset_wb,[
                (('lowres_surf_lh',utility.select,0),'lh_surf'),
                (('lowres_surf_rh',utility.select,0),'rh_surf'),
                ('lowres_rois_coords','sc_coords')]),
        ])
        

    n_group_hemi_surfs = pe.Node(
        utility.Merge(2),
        run_without_submitting=True,
        name='group_hemi_surfs')

    n_st_realign = pe.MapNode(
        nipy.SpaceTimeRealigner(
            tr=tr,
            slice_info=2,
            slice_times='ascending'),
        iterfield=['in_file'],
        name='st_realign')
    n_st_realign.plugin_args = high_mem_queue_args

    n_bbreg_epi = pe.MapNode(
        freesurfer.BBRegister(
            contrast_type='t2',
            reg_frame=0,
            registered_file=True,
            init='fsl'),
        iterfield=['source_file'],
        name='bbreg_epi')

    n_mask2epi= pe.Node(
        freesurfer.ApplyVolTransform(interp='nearest',inverse=True),
        name='mask2epi')

    n_epi2t1_bbreg2xfm = pe.MapNode(
        freesurfer.preprocess.Tkregister(
            xfm_out='fmap2t1.xfm',
            freeview='freeview.mat',
            fsl_reg_out='fsl_reg_out.mat',
            lta_out='lta.mat',
            no_edit=True),
        iterfield=['reg_file','mov'],
        run_without_submitting=True,
        name='epi2t1_bbreg2xfm')

    n_epi2t1_xfm2mat = pe.MapNode(
        utility.Function(
            input_names = ['xfm'],
            output_names = ['mat'],
            function=xfm2mat),
        iterfield=['xfm'],
        run_without_submitting=True,
        name='epi2t1_xfm2mat')

    n_repeat_fieldmaps = pe.Node(
        utility.Function(
            input_names = ['fmri_scans','fieldmaps','arg_names','fieldmap_regs'],
            output_names = ['fieldmaps','fieldmap_regs'],
            function = repeat_fieldmaps),
        run_without_submitting=True,
        name='repeat_fieldmaps')
    n_repeat_fieldmaps.inputs.arg_names = n_repeat_fieldmaps.interface._input_names[3:]

    n_convert_motion_par = generic_pipelines.fmri_surface.n_convert_motion_par

    n_moco_noco = pe.MapNode(
        nipy.preprocess.OnlineRealign(
            iekf_jacobian_epsilon = 1e-2,
            iekf_convergence = 1e-2,
            iekf_max_iter = 8,
            iekf_min_nsamples_per_slab = 32,
            iekf_observation_var = 1e6,
            iekf_transition_cov = 1e-3,
            iekf_init_state_cov = 1e-1,
            echo_time=echo_time,
            echo_spacing=echo_spacing * 2 * np.pi, # topup field in Hz * 2pi = rad/s
            phase_encoding_dir=phase_encoding_dir,
            middle_surface_position = middle_surface_position,
            register_gradient=False,
            bias_correction=True,
            bias_sigma=15,
            interp_rbf_sigma=1.5,
            interp_cortical_anisotropic_kernel=True,
            resampled_first_frame='frame1.nii',
            out_file_format='ts.h5',
            fieldmap_recenter_values=False,
            fieldmap_unmask=True),
        iterfield = ['dicom_files','fieldmap','fieldmap_reg','init_reg'],
        overwrite=False,
        name = 'moco_noco_biascorr')
    #not really using much memory, but better with multicore blas 
    n_moco_noco.plugin_args = pe_queue_args

    n_moco_bc_mvpa = n_moco_noco.clone('moco_bc_mvpa_aniso_new')
    n_moco_mvpa = n_moco_bc_mvpa.clone('moco_mvpa_aniso_new')
    n_moco_mvpa.inputs.bias_correction = False
    ### bug with cloning of mapnode
    n_moco_mvpa.interface.inputs.bias_correction = False    


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
    
    w.base_dir = proc_dir
    w.connect([
        (si, n_anat_grabber, [('subject_id',)*2]),
        
        (n_anat_grabber, n_group_hemi_surfs,[
            (('lowres_surf_lh',name_struct,'CORTEX_LEFT'),'in1'),
            (('lowres_surf_rh',name_struct,'CORTEX_RIGHT'),'in2')]),
        
        (w.get_node('all_func_dirs'),n_fmri_convert,[(('fmri_all',flatten_remove_none),'dicom_files')]),
        ])
    """
        #(n_fmri_convert,n_st_realign,[('nifti_file','in_file')]),
        (n_anat_grabber,n_bbreg_epi,[('subjects_dir',)*2]),
        (si,n_bbreg_epi,[(('subject_id',wrap(str),[]),'subject_id')]),
        (n_st_realign,n_bbreg_epi,[('out_file','source_file')]),
                
        (n_st_realign,n_epi2t1_bbreg2xfm,[('out_file','mov')]),
        (n_bbreg_epi, n_epi2t1_bbreg2xfm,[('out_reg_file','reg_file')]),
        (n_anat_grabber,n_epi2t1_bbreg2xfm,[('subjects_dir',)*2]),
        (n_epi2t1_bbreg2xfm, n_epi2t1_xfm2mat,[('xfm_out','xfm')]),
        
        #(n_epi2t1_xfm2mat,n_convert_motion_par,[('mat','epi2t1')]),
        #(n_st_realign, n_convert_motion_par,[('par_file','motion')]),
        #(n_fmri_convert, n_convert_motion_par,[('nifti_file','matrix_file')]),
        
        (w.get_node('all_func_dirs'),n_repeat_fieldmaps,[('fmri_all','fmri_scans')]),
        (n_fieldmap_fmri_convert, n_repeat_fieldmaps, [('fieldmap_file','fieldmaps')]),
        (n_xfm2mat, n_repeat_fieldmaps, [('mat','fieldmap_regs')]),
        
        ])
    """

    n_dataset_mvpa_moco = pe.Node(
        CreateDataset(tr=tr,
                      behavioral_data_path=os.path.join(data_dir,'Behavior'),
                      design=os.path.join(project_dir,'data/mvpa_only.csv'),
                      #design=os.path.join(project_dir,'data/design.csv'),
                      detrend=False,
                      median_divide=False,
                      wavelet_despike=False,
                      interp_bad_tss=False),
        name='dataset_mvpa_moco')
    n_dataset_mvpa_moco.plugin_args = high_mem_queue_args
    n_dataset_mvpa_moco_bc = n_dataset_mvpa_moco.clone('dataset_mvpa_moco_bc_hptf')
    n_dataset_mvpa_moco_bc.inputs.hptf=True
    n_dataset_mvpa_moco_bc.inputs.hptf_thresh=4
    
    n_dataset_newmoco = n_dataset_mvpa_moco_bc.clone('dataset_moco_bc')
    n_dataset_newmoco.inputs.design = os.path.join(project_dir,'data/design.csv')

    moco_noco = False
    if moco_noco:
        w.connect([
            (w.get_node('all_func_dirs'), n_moco_noco,[(('fmri_ap_all', flatten_remove_none),'dicom_files')]),
            
            (n_anat_grabber, n_moco_noco,[
                ('norm','surfaces_volume_reference'),
                ('cropped_mask','mask'),
                #(('pve_maps', utility.select, -1),'gm_pve'),
                (('pve_maps', utility.select, 2),'wm_pve'),
                (('lowres_rois_coords',name_struct,'SUBCORTICAL_CEREBELLUM',
                  '/home/bpinsard/data/projects/motion_correction/code/aparc.a2009s+aseg_subcortical_subset.txt'),
                 'resample_rois')]),
            (n_group_hemi_surfs, n_moco_noco,[('out','resample_surfaces')]),

            (n_topup, n_moco_noco, [('out_field','fieldmap')]),
            (n_topup2t1_mat, n_moco_noco, [('mat','fieldmap_reg'),
                                           ('mat','init_reg')]),
            # remove fieldmap for topup
            #(n_repeat_fieldmaps, n_moco_noco,[
            #('fieldmaps','fieldmap'),
            #('fieldmap_regs','fieldmap_reg')]),

            (si,n_dataset_newmoco,[('subject_id',)*2]),
            (n_moco_noco,n_dataset_newmoco,[('out_file','ts_files')]),
            (w.get_node('all_func_dirs'),n_dataset_newmoco,[(('fmri_all',flatten_remove_none),'dicom_dirs')]),

        ])

    def select_mvpa(l):
        if isinstance(l, list) and len(l)>0:
            if isinstance(l[-1], list) and len(l[-1])>1:
                return l[-1][-2:]
            return l[-2:]
        return []

    moco_noco_mvpa = False
    if moco_noco_mvpa:
        for n in [n_moco_bc_mvpa]:#,n_moco_mvpa]:
            w.connect([
                (w.get_node('all_func_dirs'), n,[(('fmri_ap_all', select_mvpa),'dicom_files')]),
                
                (n_anat_grabber, n,[
                    ('norm','surfaces_volume_reference'),
                    ('cropped_mask','mask'),
                    #(('pve_maps', utility.select, -1),'gm_pve'),
                    (('pve_maps', utility.select, 2),'wm_pve'),
                    (('lowres_rois_coords',name_struct,'SUBCORTICAL_CEREBELLUM',
                      '/home/bpinsard/data/projects/motion_correction/code/aparc.a2009s+aseg_subcortical_subset.txt'),
                     'resample_rois')]),
                (n_group_hemi_surfs, n,[('out','resample_surfaces')]),

                (n_topup, n, [(('out_field',utility.select,slice(-2,None)),'fieldmap')]),
                (n_topup2t1_mat, n, [(('mat',utility.select,slice(-2,None)),'fieldmap_reg'),
                                     (('mat',utility.select,slice(-2,None)),'init_reg')]),
            ])
        w.connect([
            (si, n_dataset_mvpa_moco, [('subject_id',)*2]),
            (n_moco_mvpa, n_dataset_mvpa_moco, [('out_file','ts_files')]),
            (w.get_node('all_func_dirs'), n_dataset_mvpa_moco, [(('fmri_ap_all',select_mvpa),'dicom_dirs')]),

            (si, n_dataset_mvpa_moco_bc, [('subject_id',)*2]),
            (n_moco_bc_mvpa, n_dataset_mvpa_moco_bc, [('out_file','ts_files')]),
            (w.get_node('all_func_dirs'), n_dataset_mvpa_moco_bc, [(('fmri_ap_all',select_mvpa),'dicom_dirs')]),
            
        ])
        eval_moco = False
        if eval_moco:
            n_corr_motion_wb = pe.MapNode(
                utility.Function(
                    input_names = ['lh_ctx_file','rh_ctx_file','sc_file','motion_file'],
                    output_names = ['corr'],
                    function=moco_eval.corr_delta_motion_wb),
                iterfield=['lh_ctx_file','rh_ctx_file','sc_file','motion_file'],
                name='cov_motion_wb')
            
            n_corr_motion_moco = pe.MapNode(
                utility.Function(
                    input_names = ['in_file','motion_file','nslabs'],
                    output_names = ['corr'],
                function=moco_eval.corr_delta_motion_moco),
                iterfield=['in_file','motion_file'],
                name='cov_motion_moco') 
            n_corr_motion_moco.inputs.nslabs = 40
            
            n_reg_motion_wb = pe.MapNode(
                utility.Function(
                    input_names = ['lh_ctx_file','rh_ctx_file','sc_file','motion_file'],
                    output_names = ['corr'],
                    function=moco_eval.reg_delta_motion_wb),
                iterfield=['lh_ctx_file','rh_ctx_file','sc_file','motion_file'],
                name='reg_motion_wb')
            
            n_reg_motion_moco = pe.MapNode(
                utility.Function(
                    input_names = ['in_file','motion_file','nslabs'],
                    output_names = ['corr'],
                    function=moco_eval.reg_delta_motion_moco),
                iterfield=['in_file','motion_file'],
                name='reg_motion_moco') 
            n_reg_motion_moco.inputs.nslabs = 40
        
            n_ddiff_var_wb = pe.MapNode(
                utility.Function(
                    input_names = ['lh_ctx_file','rh_ctx_file','sc_file'],
                    output_names = ['corr'],
                    function=moco_eval.ddiff_var_wb),
                iterfield=['lh_ctx_file','rh_ctx_file','sc_file'],
                name='ddiff_var_wb')

            n_ddiff_var_moco = pe.MapNode(
                utility.Function(
                    input_names = ['in_file'],
                    output_names = ['corr'],
                    function=moco_eval.ddiff_var_moco),
                iterfield=['in_file'],
                name='ddiff_var_moco')

            w.connect([
                #cov/corr
                (wb_pipe, n_corr_motion_wb,[('volume2surface_lh.out_file','lh_ctx_file'),
                                            ('volume2surface_rh.out_file','rh_ctx_file'),
                                            ('volume2surface_sc.out_file','sc_file')]),
                (n_mcflirt, n_corr_motion_wb,[('par_file','motion_file')]),
                
                (n_moco_bc_mvpa, n_corr_motion_moco, [('out_file','in_file'),
                                                      ('motion_params','motion_file')]),
                # betas regs
                (wb_pipe, n_reg_motion_wb,[('volume2surface_lh.out_file','lh_ctx_file'),
                                           ('volume2surface_rh.out_file','rh_ctx_file'),
                                           ('volume2surface_sc.out_file','sc_file')]),
                (n_mcflirt, n_reg_motion_wb,[('par_file','motion_file')]),
                
                (n_moco_bc_mvpa, n_reg_motion_moco, [('out_file','in_file'),
                                                     ('motion_params','motion_file')]),
                # ddiff 
                (wb_pipe, n_ddiff_var_wb,[('volume2surface_lh.out_file','lh_ctx_file'),
                                          ('volume2surface_rh.out_file','rh_ctx_file'),
                                          ('volume2surface_sc.out_file','sc_file')]),
            
                (n_moco_bc_mvpa, n_ddiff_var_moco, [('out_file','in_file')])
                
            ])

    n_dataset_noisecorr = pe.Node(
        CreateDataset(tr=tr,
                      behavioral_data_path=os.path.join(data_dir,'Behavior'),
                      design=os.path.join(project_dir,'data/design.csv')),
        name='dataset_noisecorr')
    n_dataset_noisecorr.plugin_args = high_mem_queue_args

    n_dataset_nofilt = pe.Node(
        CreateDataset(tr=tr,
                      behavioral_data_path=os.path.join(data_dir,'Behavior'),
                      design=os.path.join(project_dir,'data/mvpa_only.csv'),
                      #design=os.path.join(project_dir,'data/design.csv'),
                      median_divide=True,
                      wavelet_despike=True,
                      interp_bad_tss=True),
        name='dataset_mvpa_wd_interp_hrf_gam1')

    n_dataset_nofilt = pe.Node(
        CreateDataset(tr=tr,
                      behavioral_data_path=os.path.join(data_dir,'Behavior'),
                      #design=os.path.join(project_dir,'data/mvpa_only.csv'),
                      design=os.path.join(project_dir,'data/design.csv'),
                      median_divide=True,
                      wavelet_despike=True,
                      interp_bad_tss=True),
        name='dataset_wd_interp_hrf_gam2')

    n_dataset_nofilt.plugin_args = high_mem_queue_args
    n_dataset_smoothed = n_dataset_noisecorr.clone('dataset_smoothed')

    nofilt_resample = False
    if nofilt_resample:

        if use_topup_fieldmap:
            n_surf_resample = n_surf_resample.clone('surf_resample_topup')
            w.connect([
                (n_repeat_topup, n_surf_resample,[('fieldmaps','fieldmap'),
                                                  ('reg_file','fieldmap_reg')]),
            ])
        else:
            w.connect([
            (n_repeat_fieldmaps, n_surf_resample,[
                    ('fieldmaps','fieldmap'),
                    ('fieldmap_regs','fieldmap_reg')]),
            ])
        
        w.connect([
            (n_anat_grabber, n_surf_resample,[
                    ('norm','surfaces_volume_reference'),
                    ('cropped_mask','mask'),
                    (('lowres_rois_coords',name_struct,'SUBCORTICAL_CEREBELLUM',
                      '/home/bpinsard/data/projects/motion_correction/code/aparc.a2009s+aseg_subcortical_subset.txt'),
                     'resample_rois'),
                    ]),

            (n_convert_motion_par, n_surf_resample,[('motion','motion')]),
            (w.get_node('all_func_dirs'), n_surf_resample,[(('fmri_all', flatten_remove_none),'dicom_files')]),
            (n_group_hemi_surfs, n_surf_resample,[('out','resample_surfaces')]),

#            (n_surf_resample, n_smooth_bp_nofilt,[('out_file','in_file')]),            
            (si,n_dataset_nofilt,[('subject_id',)*2]),
            (n_surf_resample,n_dataset_nofilt,[('out_file','ts_files')]),
            (w.get_node('all_func_dirs'),n_dataset_nofilt,[(('fmri_all',flatten_remove_none),'dicom_dirs')]),

            ])
    noise_filt = False
    if noise_filt:
        w.connect([
            (n_convert_motion_par, n_noise_corr,[('motion','motion')]),
            (w.get_node('all_func_dirs'),n_noise_corr,[(('fmri_all', flatten_remove_none),'dicom_files')]),
            
            (n_anat_grabber, n_noise_corr,[
                ('norm','surfaces_volume_reference'),
                ('cropped_mask','mask'),
                (('lowres_rois_coords',name_struct,'SUBCORTICAL_CEREBELLUM',
                  '/home/bpinsard/data/projects/motion_correction/code/aparc.a2009s+aseg_subcortical_subset.txt'),
                 'resample_rois'),
                ('pve_maps','partial_volume_maps')]),
            (n_repeat_fieldmaps, n_noise_corr,[
                ('fieldmaps','fieldmap'),
                ('fieldmap_regs','fieldmap_reg')]),
            (n_group_hemi_surfs, n_noise_corr,[('out','resample_surfaces')]),

            (si,n_dataset_noisecorr,[('subject_id',)*2]),
            (n_noise_corr,n_dataset_noisecorr,[('out_file','ts_files')]),
            (w.get_node('all_func_dirs'),n_dataset_noisecorr,[(('fmri_all',flatten_remove_none),'dicom_dirs')]),

            #(n_noise_corr, n_smooth_bp,[('out_file','in_file')]),
            #(si,n_dataset_smoothed,[('subject_id',)*2]),
            #(n_smooth_bp,n_dataset_smoothed,[('out_file','ts_files')]),
            #(w.get_node('all_func_dirs'),n_dataset_smoothed,[(('fmri_all',flatten_remove_none),'dicom_dirs')]),

        ])

    eeg_coords = True
    if eeg_coords:
        
        n_eeg_coords_bold = pe.MapNode(
            utility.Function(
                input_names = ['in_file', 'warp_file', 'coords_file'],
                output_names = ['bold_signal'],
                function = eeg_coords_bold_signal),
            iterfield=['in_file'],
            name = 'eeg_coords_bold')
        n_eeg_coords_bold.inputs.coords_file = os.path.join(project_dir,'data','grid_eeg_loc_mni.csv')
        w.connect([
            (n_anat_grabber, n_eeg_coords_bold, [('warp2mni','warp_file')]),
            (n_apply_registration, n_eeg_coords_bold, [('out_file','in_file')])])
        
    return w 


def xfm2mat(xfm):
    import os
    import numpy as np
    mat=np.loadtxt(xfm, skiprows=5,usecols=range(4))
    np.savetxt("out.mat",np.vstack([mat,[0,0,0,1]]))
    return os.path.abspath("out.mat")
    
from nipype.interfaces.base import (TraitedSpec, BaseInterface, traits,
                                    BaseInterfaceInputSpec, isdefined, File, Directory,
                                    InputMultiPath, OutputMultiPath)


class CreateDatasetInputSpec(BaseInterfaceInputSpec):
    subject_id = traits.Int(mandatory=True)
    ts_files = traits.List(File(exists=True))
    dicom_dirs = traits.List(Directory(exists=True))
    data_path = traits.Str('FMRI/DATA',usedefault=True,)
    design = File(exists=True,mandatory=True)
    behavioral_data_path=Directory()
    mean_divide = traits.Bool(False, usedefault=True)
    median_divide = traits.Bool(False, usedefault=True)
    detrend = traits.Bool(False, usedefault=True)
    
    wavelet_despike = traits.Bool(False, usedefault=True)
    hptf = traits.Bool(False, usedefault=True)
    hptf_thresh = traits.Float(32., usedefault=True)
    interp_bad_tss = traits.Bool(False, usedefault=True)
    tr = traits.Float(mandatory=True)

class CreateDatasetOutputSpec(TraitedSpec):
    dataset = File(exists=True, mandatory=True)
    glm_dataset = File(exists=True, mandatory=True)
    glm_stim_dataset = File(exists=True, mandatory=True)

class CreateDataset(BaseInterface):

    input_spec = CreateDatasetInputSpec
    output_spec = CreateDatasetOutputSpec

    def _load_ts_file(self, ts_file, dicom_dir, tr):
        return mvpa_dataset.ds_from_ts(
            ts_file,
            tr=tr)

    def _run_interface(self, runtime):
        dss = []
        dss_glm = []
        dss_glm_stim = []
        scan_id = 0

        subject_id = self.inputs.subject_id
        used_ts_files = []

        empty_to_none =lambda x: x if len(x) else None
        design = np.atleast_2d(
            np.loadtxt(
                self.inputs.design,
                dtype=np.object,
                delimiter=',',
                converters={0:int,1:str,2:str,3:empty_to_none,4:empty_to_none,5:empty_to_none,6:bool}))
        scan_id=-1
        for day,ses_name,mri_name,seq_name,beh,scan_idx,optional in design:
            seq_idx = None
            seq_info = None
            if not seq_name is None and not beh is None:
                seq_info = SEQ_INFO
                seq_idx = [[s[0] for s in seq_info].index(seq_name)] * 14
            behavior_file = None
            if not beh is None:
                # take the last behavioral file which matches in case of failed task
                behavior_file = sorted(glob.glob(
                    os.path.join(
                        self.inputs.behavioral_data_path,
                        'CoRe_%03d_D%d/CoRe_%03d_%s_?.mat'%(subject_id,day,subject_id,beh))))
                if not len(behavior_file): 
                    if optional:
                        continue
                    else:
                        break # do not crash, in case we want to analyze for early boost...
                        #raise RuntimeError('missing data')
                behavior_file = behavior_file[-1]

            # deal with multiple scans
            select_ts = [('_D%d/'%day in dd and mri_name in dd) for dd in self.inputs.dicom_dirs]
            ts_files = [f for f,sts in zip(self.inputs.ts_files,select_ts) if sts]
            dicom_dirs = [d for d,sts in zip(self.inputs.dicom_dirs,select_ts) if sts]

            if scan_idx is not None:
                scan_idx = int(scan_idx)
                if scan_idx >= len(ts_files):
                    if optional:
                        continue
                    else:
                        break
                        #raise RuntimeError('missing data')
                ts_files = [ts_files[int(scan_idx)]]
            ts_files = [f for f in ts_files if f not in used_ts_files]

            

            for ts_file, dicom_dir in zip(ts_files, dicom_dirs):
                used_ts_files.append(ts_file)                
                #print day, ses_name, mri_name, ts_file[-12:], str(behavior_file).split('/')[-1]
                #continue

                ds = self._load_ts_file(ts_file, dicom_dir, self.inputs.tr)
                mvpa_dataset.ds_set_attributes(
                    ds,
                    behavior_file,
                    seq_info=seq_info, seq_idx=seq_idx,
                    tr=self.inputs.tr)
                mvpa_dataset.preproc_ds(
                    ds,
                    mean_divide=self.inputs.mean_divide,
                    median_divide=self.inputs.median_divide,
                    wav_despike=self.inputs.wavelet_despike,
                    hptf=self.inputs.hptf,
                    hptf_thresh=self.inputs.hptf_thresh,
                    tr=self.inputs.tr)
                if ds.nsamples <= 10:
                    break
                scan_id += 1

                if self.inputs.interp_bad_tss:
                    ds = mvpa_dataset.interp_bad_ts(ds)
                ds.sa['scan_name'] = [ses_name]*ds.nsamples
                ds.sa['scan_id'] = [scan_id]*ds.nsamples
                dss.append(ds)
                if beh is not None:
                    reg_groups = np.unique([n.split('_')[0] for n in ds.sa.regressors_exec.dtype.names])
                    glm_hptf = None
                    if self.inputs.hptf:
                        glm_hptf = self.inputs.hptf_thresh
                    ds_glm = mvpa_dataset.ds_tr2glm(ds, 'regressors_exec', 
                                                    reg_groups[reg_groups!='constant'],['constant'],
                                                    hptf=glm_hptf)
                    dss_glm.append(ds_glm)
                    reg_groups = np.unique([n.split('_')[0] for n in ds.sa.regressors_stim.dtype.names])
                    ds_glm = mvpa_dataset.ds_tr2glm(ds, 'regressors_stim',
                                                    reg_groups[reg_groups!='constant'],['constant'])
                    dss_glm_stim.append(ds_glm)

                    for a in ['regressors_exec','regressors_stim','regressors_exec_evt',
                              'regressors_blocks','regressors_stim_evt']:
                        if a in ds.sa:
                            del ds.sa[a]
                # used ts files to avoid repeating
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
        mvpa_dataset.add_aparc_ba_fa(
            ds, self.inputs.subject_id,
            pproc_tpl=os.path.join(proc_dir,'core_sleep','surface_32k','_subject_id_%s'))
        ds_glm.fa = ds.fa
        ds_glm_stim.fa = ds.fa
        
        outputs = self._list_outputs()
        print 'saving'
        ds.save(outputs['dataset'])
        ds_glm.save(outputs['glm_dataset'])
        ds_glm_stim.save(outputs['glm_stim_dataset'])

        print 'completed'
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['dataset'] = os.path.abspath('./ds_%s.h5'%self.inputs.subject_id)
        outputs['glm_dataset'] = os.path.abspath('./glm_ds_%s.h5'%self.inputs.subject_id)
        outputs['glm_stim_dataset'] = os.path.abspath('./glm_stim_ds_%s.h5'%self.inputs.subject_id)
        return outputs



class CreateDatasetWBInputSpec(CreateDatasetInputSpec):
    ts_files = traits.List(traits.Tuple(*([File(exists=True)]*3)))

    lh_surf = File(exists=True)
    rh_surf = File(exists=True)
    sc_coords = File(exists=True)

class CreateDatasetWB(CreateDataset):

    input_spec = CreateDatasetWBInputSpec
    output_spec = CreateDatasetOutputSpec

    def _load_ts_file(self, ts_file, dicom_dir, tr):
        tss = np.hstack([np.asarray([da.data for da in nb.load(f).darrays]) for f in ts_file])
        ds = mvpa2.datasets.Dataset(tss)
        
        dcm = dicom.read_file(sorted(glob.glob(os.path.join(dicom_dir, '*')))[0])
        dt = datetime.datetime.strptime(dcm.AcquisitionDate+':'+dcm.AcquisitionTime,'%Y%m%d:%H%M%S.%f')
        tstp = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
        ds.sa['time'] = tstp + np.arange(ds.nsamples)*tr

        lh_gii = nb.load(self.inputs.lh_surf)
        rh_gii = nb.load(self.inputs.rh_surf)
        sc_coords = np.loadtxt(self.inputs.sc_coords, delimiter=',')
        ds.fa['coordinates'] = np.vstack([
            lh_gii.darrays[0].data,
            rh_gii.darrays[0].data,
            sc_coords[:,:3]])
        ds.a['triangles'] = np.vstack([
            lh_gii.darrays[1].data,
            rh_gii.darrays[1].data + len(lh_gii.darrays[0].data)])

        ds.fa['voxel_indices'] = np.empty((ds.nfeatures,3), dtype=np.int)
        ds.fa.voxel_indices.fill(-1)
        rois_offset = len(lh_gii.darrays[0].data) + len(rh_gii.darrays[0].data)
        ds.fa.voxel_indices[rois_offset:] = sc_coords[:,3:6]
        ds.fa['node_indices'] = np.arange(ds.nfeatures, dtype=np.uint)

        return ds

def repeat_fieldmaps(fmri_scans, fieldmaps, arg_names, **kwargs):
    from datetime import datetime
    import glob, os, re
    import dicom
    import numpy as np
    fmaps_out = []
    args_out = dict([(a,[]) for a in arg_names])
    fmap_regs_out = []
    i = 0
    fieldmap_datetimes = [datetime.strptime(re.search('[0-9]{8}_[0-9]{6}',fmap).group(0),'%Y%m%d_%H%M%S') \
                          for fmap in fieldmaps]
    for sess in fmri_scans:
        if sess != None:
            for scan in sess:
                dcm = dicom.read_file(glob.glob(os.path.join(scan,'*0001.dcm'))[0])
                fmri_time = datetime.strptime(dcm.AcquisitionDate+dcm.AcquisitionTime,'%Y%m%d%H%M%S.%f')
                time_to_fmap = np.asarray([(fmap_time-fmri_time).total_seconds() for fmap_time in fieldmap_datetimes])
                fmap_idx = np.where(time_to_fmap>0)[0]
                fmap_idx = fmap_idx[0] if len(fmap_idx) else -1
                if time_to_fmap[fmap_idx] > 3600*6: #more than 6 hours later!! only for PA usually
                    fmap_idx = fmap_idx-1
                fmaps_out.append(fieldmaps[fmap_idx])
                for argn,argv in kwargs.items():
                    args_out[argn].append(argv[fmap_idx])
                i += 1
    return (fmaps_out,)+tuple([args_out[argn] for argn in arg_names])


def preproc_eeg():
    
    w = dicom_dirs()
    templates = dict(
        subjects_dir=[['t1_preproc/_subject_id','subject_id','freesurfer']],
        norm = [['t1_preproc/_subject_id','subject_id','freesurfer/*[!e]/mri/norm.mgz']],
        white_matter_surface = [['extract_wm_surface/_subject_id','subject_id','surf_decimate/rlh.aparc+aseg_wm.nii_smoothed_mask.all']],
        cropped_mask = [['t1_preproc/_subject_id','subject_id','autobox_mask_fs/*.nii.gz']],
        cropped_t1 = [['t1_preproc/_subject_id','subject_id','crop_t1/*.nii.gz']],
        pve_maps = [
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.cortex.nii.gz'],
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.subcort_gm.nii.gz'],
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.wm.nii.gz'],
            ['t1_preproc/_subject_id','subject_id','compute_pvmaps/*.csf.nii.gz']],
        lowres_surf_lh = [
            ['surface_32k/_subject_id','subject_id',
             'white_resample_surf/mapflow/_white_resample_surf0/lh.white_converted.32k.gii'],
            ['surface_32k/_subject_id','subject_id',
             'pial_resample_surf/mapflow/_pial_resample_surf0/lh.pial_converted.32k.gii']],
        lowres_surf_rh = [
            ['surface_32k/_subject_id','subject_id',
             'white_resample_surf/mapflow/_white_resample_surf1/rh.white_converted.32k.gii'],
            ['surface_32k/_subject_id','subject_id',
             'pial_resample_surf/mapflow/_pial_resample_surf1/rh.pial_converted.32k.gii']],
        lowres_rois_coords = [['ants_for_subcortical/_subject_id','subject_id','coords_itk2nii/atlas_coords_nii.csv']],
        )
    n_anat_grabber = pe.Node(
        nio.DataGrabber(
            infields=['subject_id'],
            outfields=templates.keys(),
            sort_filelist=True,
            raise_on_empty=False,
            base_directory = proc_dir),
        run_without_submitting=True,
        name='anat_grabber')
    n_anat_grabber.inputs.template = 'core_sleep/%s_%s/%s'
    n_anat_grabber.inputs.template_args = templates


    n_convert_scout_dicom = pe.MapNode(
        np_dcmstack.DCMStackAnatomical(
            meta_force_add=meta_tag_force,
            out_file_format = 'scout'+file_pattern),
        iterfield=['dicom_files'],
        name='convert_scout_dicoms')
    
    w.connect(
        (w.get_node('all_func_dirs'), n_convert_scout_dicom,[(('aa_scout_all',flatten_remove_none),'dicom_files')]),
    )

    return w


def coords2fakesurf(in_file):
    import os
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    subcoords = np.loadtxt(in_file, delimiter=',', usecols=(0,1,2))
    points_da = nb.gifti.GiftiDataArray(subcoords[:,:3].astype(np.float32), 'pointset')
    points_da.ext_offset = ''
    tris_da = nb.gifti.GiftiDataArray(np.arange(6,dtype=np.int32).reshape(-1,3),'triangle')
    tris_da.ext_offset = ''
    fake_surf = nb.gifti.GiftiImage(darrays=[points_da, tris_da])
    out_fname = os.path.abspath(fname_presuffix(in_file, newpath='./', suffix='.gii', use_ext=False))
    nb.save(fake_surf, out_fname)
    return out_fname
    
    
    

def eeg_coords_bold_signal(in_file, warp_file, coords_file):
    import os, sys
    sys.path.insert(0, '/home/bpinsard/data/projects/CoRe')
    import numpy as np
    import scipy.io
    from scipy.ndimage import map_coordinates
    import nibabel as nb
    import dipy.align.vector_fields as vfu
    import core.mvpa.dataset
    from nipype.utils.filemanip import fname_presuffix

    coords = np.loadtxt(coords_file, delimiter=',', usecols=(0,1,2))
    nii = nb.load(in_file)

    warp = np.load(warp_file)
    backward = warp['backward']
    prealign = warp['prealign']
    postalign = np.linalg.inv(prealign.dot(nii.affine))
    shift = vfu.interpolate_vector_3d(backward, coords[:,:3])[0]
    coords[:,:3] += shift
    coords[:,:3] = nb.affines.apply_affine(postalign, coords[:,:3])
    
    data = nii.get_data().astype(np.float32)

    out_data = np.empty((coords.shape[0],nii.shape[3]), dtype=data.dtype)
    for v in range(nii.shape[3]):
        map_coordinates(data[...,v],coords.T,out_data[:,v])
    ds = core.mvpa.dataset.Dataset(out_data)
    core.mvpa.dataset.preproc_ds(ds, detrend=True)

    out_fname = os.path.abspath(fname_presuffix(in_file, newpath='./', suffix='.mat', use_ext=False))    
    scipy.io.savemat(out_fname, dict(bold=ds.samples))
    return out_fname
    
    
