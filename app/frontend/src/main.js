import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import PlayableApp from './PlayableApp.vue'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { faRedo, faLocationArrow, faPaperPlane, faBullseye, faToggleOn, faToggleOff, faKeyboard, faChevronLeft, faChevronRight, faStepForward, faArrowLeft, faArrowRight, faCamera, faPlay } from '@fortawesome/free-solid-svg-icons'
import { faHandPointer } from '@fortawesome/free-regular-svg-icons'

library.add(faRedo, faLocationArrow, faPaperPlane, faBullseye, faHandPointer, faToggleOn, faToggleOff, faKeyboard, faChevronLeft, faChevronRight, faStepForward, faArrowLeft, faArrowRight, faCamera, faPlay)

const appMode = String(import.meta.env?.VITE_APP_MODE || '').trim().toLowerCase()
const RootComponent = appMode === 'playable' ? PlayableApp : App

const app = createApp(RootComponent)
app.component('font-awesome-icon', FontAwesomeIcon)
app.mount('#app')
